#!/usr/bin/env python3
"""
Lightweight constant-velocity 2D Kalman filter.

This module intentionally depends only on NumPy (no SciPy, no filterpy) so it
stays cheap enough to run inside the perception loop on a CPU-only robot.

State vector (4x1):
    [x, y, vx, vy]^T

Measurement vector (2x1):
    [x, y]^T  (normalized bounding-box center in [0, 1])

The motion model is a discrete constant-velocity model. Process noise is
injected through a white-acceleration model, which lets the filter tolerate
small accelerations (a person changing speed) without needing an explicit
acceleration state.
"""

import numpy as np


class KalmanFilter2D:
    """Constant-velocity 2D Kalman filter for target bounding-box centers.

    Coordinates are expected to be normalized image coordinates in [0, 1],
    but the filter is unit-agnostic as long as ``dt`` and the noise values use
    consistent units.
    """

    def __init__(self,
                 process_noise: float = 1e-1,
                 measurement_noise: float = 2e-3,
                 dt: float = 1.0 / 30.0,
                 initial_covariance: float = 1.0,
                 max_coast_frames: int = 15):
        """Create a filter.

        Args:
            process_noise: White-acceleration spectral density (q). Larger
                values make the filter trust measurements more (less lag,
                more jitter).
            measurement_noise: Measurement variance (r). Larger values make
                the filter trust the motion model more (more smoothing).
            dt: Time between frames in seconds.
            initial_covariance: Initial uncertainty for every state entry.
            max_coast_frames: Maximum number of consecutive predict-only
                steps allowed during an occlusion.
        """
        if dt <= 0.0:
            raise ValueError('dt must be positive')
        if process_noise < 0.0:
            raise ValueError('process_noise must be non-negative')
        if measurement_noise <= 0.0:
            raise ValueError('measurement_noise must be positive')
        if initial_covariance <= 0.0:
            raise ValueError('initial_covariance must be positive')
        if max_coast_frames < 0:
            raise ValueError('max_coast_frames must be non-negative')

        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.dt = float(dt)
        self.initial_covariance = float(initial_covariance)
        self.max_coast_frames = int(max_coast_frames)

        # Constant-velocity state transition matrix.
        self.F = np.array([
            [1.0, 0.0, self.dt, 0.0],
            [0.0, 1.0, 0.0, self.dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        # Measurement matrix: we observe position only.
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ])

        # Discrete white-acceleration process noise:
        #   G = [[dt^2/2, 0], [0, dt^2/2], [dt, 0], [0, dt]]
        #   Q = q * G @ G.T
        G = np.array([
            [0.5 * self.dt ** 2, 0.0],
            [0.0, 0.5 * self.dt ** 2],
            [self.dt, 0.0],
            [0.0, self.dt],
        ])
        self.Q = self.process_noise * (G @ G.T)

        # Measurement covariance (position x and y are measured independently).
        self.R = self.measurement_noise * np.eye(2)
        self._identity = np.eye(4)

        self.reset()

    @property
    def is_initialized(self) -> bool:
        """Whether at least one measurement has been processed."""
        return self._initialized

    @property
    def state(self) -> np.ndarray:
        """Copy of the current [x, y, vx, vy] state."""
        return self._state.copy()

    @property
    def position(self):
        """Current (x, y) estimate, or None before initialization."""
        if not self._initialized:
            return None
        return self._state[:2].copy()

    @property
    def velocity(self):
        """Current (vx, vy) estimate, or None before initialization."""
        if not self._initialized:
            return None
        return self._state[2:].copy()

    @property
    def coast_frames(self) -> int:
        """Consecutive predict-only steps since the last measurement."""
        return self._coast_frames

    def reset(self):
        """Clear the estimate and covariance (e.g. after losing the track)."""
        self._state = np.zeros(4)
        self._covariance = np.eye(4) * self.initial_covariance
        self._initialized = False
        self._coast_frames = 0

    def update(self, x: float, y: float) -> np.ndarray:
        """Predict one step, then correct with a measured center.

        Returns:
            The updated (x, y) estimate as a NumPy array.
        """
        measurement = np.array([float(x), float(y)])

        if not self._initialized:
            # Seed position from the first measurement; velocity is unknown.
            self._state = np.array([measurement[0], measurement[1], 0.0, 0.0])
            self._initialized = True
            self._coast_frames = 0
            return self.position

        self._predict_step()

        innovation = measurement - self.H @ self._state
        innovation_covariance = self.H @ self._covariance @ self.H.T + self.R
        kalman_gain = (
            self._covariance
            @ self.H.T
            @ np.linalg.inv(innovation_covariance)
        )

        self._state = self._state + kalman_gain @ innovation

        # Joseph form keeps the covariance symmetric and positive semi-definite.
        i_kh = self._identity - kalman_gain @ self.H
        self._covariance = (
            i_kh @ self._covariance @ i_kh.T
            + kalman_gain @ self.R @ kalman_gain.T
        )
        self._coast_frames = 0
        return self.position

    def predict(self):
        """Advance the state without a measurement.

        Used during brief occlusions. After ``max_coast_frames`` predict-only
        steps, the estimate is frozen so a stale motion model cannot diverge
        indefinitely.

        Returns:
            The predicted (x, y), or None if the filter has no state yet.
        """
        if not self._initialized:
            return None

        if self._coast_frames < self.max_coast_frames:
            self._predict_step()
            self._coast_frames += 1

        return self.position

    def _predict_step(self):
        """Apply one constant-velocity prediction."""
        self._state = self.F @ self._state
        self._covariance = (
            self.F @ self._covariance @ self.F.T + self.Q
        )
