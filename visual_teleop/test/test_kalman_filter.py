#!/usr/bin/env python3
"""
Unit tests for the constant-velocity 2D Kalman filter.

These tests are pure NumPy (no ROS, camera, or model inference) and verify
both numerical behaviour and the jitter-reduction claim against the previous
5-frame moving-average smoother.
"""

import unittest

import numpy as np

from visual_teleop.utils.kalman_filter import KalmanFilter2D


def moving_average(track: np.ndarray, window: int = 5) -> np.ndarray:
    """Causal moving average used by the previous smoothing implementation."""
    return np.array([
        track[max(0, i - window + 1):i + 1].mean(axis=0)
        for i in range(len(track))
    ])


def jitter(track: np.ndarray) -> float:
    """RMS magnitude of the second difference (frame-to-frame acceleration).

    Lower is smoother. Using the second difference removes constant-velocity
    motion, so this isolates high-frequency bbox jitter.
    """
    second_difference = np.diff(track, n=2, axis=0)
    return float(np.sqrt(np.mean(np.sum(second_difference ** 2, axis=1))))


class TestKalmanFilter2D(unittest.TestCase):
    """Numerical and trajectory-smoothness tests for KalmanFilter2D."""

    def test_first_measurement_initializes_state(self):
        """The first measurement seeds position with zero initial velocity."""
        kf = KalmanFilter2D()

        self.assertFalse(kf.is_initialized)
        self.assertIsNone(kf.position)
        self.assertIsNone(kf.velocity)

        estimate = kf.update(0.25, 0.75)

        self.assertTrue(kf.is_initialized)
        np.testing.assert_allclose(estimate, [0.25, 0.75], atol=1e-12)
        np.testing.assert_allclose(kf.velocity, [0.0, 0.0], atol=1e-12)
        self.assertEqual(kf.coast_frames, 0)

    def test_constant_velocity_tracking_converges(self):
        """A noise-free constant-velocity track should be tracked accurately."""
        # dt=1 keeps velocity in "per-frame" units for an easy assertion.
        kf = KalmanFilter2D(process_noise=1e-3, measurement_noise=1e-4, dt=1.0)
        true_vx, true_vy = 0.004, -0.002

        for i in range(60):
            kf.update(0.2 + true_vx * i, 0.8 + true_vy * i)

        np.testing.assert_allclose(
            kf.velocity, [true_vx, true_vy], atol=2e-3,
            err_msg='Velocity estimate did not converge to the true velocity')
        np.testing.assert_allclose(
            kf.position, [0.2 + true_vx * 59, 0.8 + true_vy * 59], atol=2e-3)

    def test_jitter_reduction_vs_moving_average(self):
        """KF should reduce jitter and RMSE versus the 5-frame moving average.

        The synthetic track is a constant-velocity person trajectory with
        Gaussian bbox-center noise (sigma = 0.01 normalized = ~6 px at 640 px).
        """
        rng = np.random.default_rng(2024)
        frames = 240
        t = np.arange(frames)
        ground_truth = np.column_stack((
            0.2 + 0.002 * t,
            0.3 + 0.001 * t,
        ))
        measurements = ground_truth + rng.normal(0.0, 0.01, ground_truth.shape)

        kf = KalmanFilter2D()
        kalman_track = np.array([kf.update(x, y) for x, y in measurements])
        average_track = moving_average(measurements, window=5)

        kalman_jitter = jitter(kalman_track)
        average_jitter = jitter(average_track)
        reduction = (1.0 - kalman_jitter / average_jitter) * 100.0

        kalman_rmse = float(np.sqrt(np.mean(np.sum(
            (kalman_track - ground_truth) ** 2, axis=1))))
        average_rmse = float(np.sqrt(np.mean(np.sum(
            (average_track - ground_truth) ** 2, axis=1))))

        # Measured on this exact seeded track: ~30.0% overall jitter reduction
        # and ~51.9% after the filter's 5-frame warm-up.
        self.assertGreater(
            reduction, 25.0,
            msg=f'Expected >25% jitter reduction, got {reduction:.1f}%')
        self.assertLess(
            kalman_rmse, average_rmse,
            msg=(f'KF RMSE {kalman_rmse:.5f} should beat moving-average '
                 f'RMSE {average_rmse:.5f}'))

        steady_state_reduction = (
            1.0 - jitter(kalman_track[5:]) / jitter(average_track[5:])
        ) * 100.0
        self.assertGreater(
            steady_state_reduction, 50.0,
            msg=(f'Expected >50% steady-state jitter reduction, got '
                 f'{steady_state_reduction:.1f}%'))

    def test_prediction_during_occlusion(self):
        """Predict-only steps should continue a constant-velocity trajectory."""
        kf = KalmanFilter2D(process_noise=1e-3, measurement_noise=1e-4)
        vx, vy = 0.004, 0.002
        last_index = 29

        for i in range(last_index + 1):
            kf.update(0.2 + vx * i, 0.4 + vy * i)

        predicted = []
        for _ in range(5):
            predicted.append(kf.predict())

        self.assertEqual(kf.coast_frames, 5)
        for step, estimate in enumerate(predicted, start=1):
            expected = np.array([
                0.2 + vx * (last_index + step),
                0.4 + vy * (last_index + step),
            ])
            np.testing.assert_allclose(
                estimate, expected, atol=2e-3,
                err_msg=f'Prediction {step} did not continue the trajectory')

    def test_coast_limit_freezes_estimate(self):
        """After max_coast_frames, predictions stop advancing (anti-divergence)."""
        kf = KalmanFilter2D(max_coast_frames=3)
        for i in range(10):
            kf.update(0.1 + 0.01 * i, 0.5)

        for _ in range(3):
            kf.predict()
        frozen_position = kf.position.copy()
        frozen_coast_count = kf.coast_frames

        for _ in range(10):
            kf.predict()

        np.testing.assert_allclose(kf.position, frozen_position)
        self.assertEqual(kf.coast_frames, frozen_coast_count)
        self.assertEqual(frozen_coast_count, 3)

    def test_update_after_occlusion_resets_coast_and_corrects(self):
        """A fresh measurement should reset the coast counter and pull the state."""
        kf = KalmanFilter2D(max_coast_frames=15)
        for i in range(10):
            kf.update(0.2 + 0.005 * i, 0.5)

        for _ in range(4):
            kf.predict()
        self.assertEqual(kf.coast_frames, 4)

        corrected = kf.update(0.30, 0.60)

        self.assertEqual(kf.coast_frames, 0)
        np.testing.assert_allclose(corrected, [0.30, 0.60], atol=0.05)

    def test_reset_clears_state(self):
        """reset() must return the filter to its pre-measurement state."""
        kf = KalmanFilter2D()
        kf.update(0.3, 0.7)
        kf.predict()

        kf.reset()

        self.assertFalse(kf.is_initialized)
        self.assertIsNone(kf.position)
        self.assertEqual(kf.coast_frames, 0)

    def test_invalid_parameters_raise(self):
        """Configuration errors should fail loudly rather than silently."""
        with self.assertRaises(ValueError):
            KalmanFilter2D(dt=0.0)
        with self.assertRaises(ValueError):
            KalmanFilter2D(process_noise=-1.0)
        with self.assertRaises(ValueError):
            KalmanFilter2D(measurement_noise=0.0)
        with self.assertRaises(ValueError):
            KalmanFilter2D(initial_covariance=0.0)
        with self.assertRaises(ValueError):
            KalmanFilter2D(max_coast_frames=-1)


if __name__ == '__main__':
    unittest.main()
