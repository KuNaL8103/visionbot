#!/usr/bin/env python3
"""
ByteTrack wrapper for visual_teleop.

Wraps supervision's ByteTrack to provide a clean interface for tracking
YOLO detections across frames.
"""

import warnings
import numpy as np
from supervision.tracker.byte_tracker.core import ByteTrack
from supervision import Detections


class TrackerWrapper:
    """
    Wrapper around supervision.ByteTrack for tracking YOLO detections.

    Provides stable track IDs across frames, handling brief occlusions
    and re-identifying targets when they reappear.
    """

    def __init__(self,
                 track_activation_threshold: float = 0.25,
                 minimum_matching_threshold: float = 0.8,
                 max_time_lost: int = 30,
                 minimum_consecutive_frames: int = 1):
        """
        Initialize ByteTrack tracker.

        Args:
            track_activation_threshold: Confidence threshold for activating new tracks
            minimum_matching_threshold: IoU threshold for matching detections to tracks
            max_time_lost: Max frames to keep lost track alive (max_age)
            minimum_consecutive_frames: Min consecutive frames to confirm a track
        """
        # Suppress the deprecation warning for cleaner logs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            # ByteTrack proxy requires positional args (v0.30.1)
            self.tracker = ByteTrack(
                track_activation_threshold,
                minimum_matching_threshold,
                max_time_lost,
                minimum_consecutive_frames
            )

    def update(self, boxes: np.ndarray, confidences: np.ndarray,
               class_ids: np.ndarray) -> Detections:
        """
        Update tracker with new detections.

        Args:
            boxes: Nx4 array in xyxy format (x1, y1, x2, y2)
            confidences: N array of confidence scores
            class_ids: N array of class IDs

        Returns:
            Detections object with track_id field populated
        """
        if len(boxes) == 0:
            # No detections - still need to update tracker to age out tracks
            detections = Detections.empty()
            return self.tracker.update_with_detections(detections)

        # Create Detections object from YOLO output
        detections = Detections(
            xyxy=boxes,
            confidence=confidences,
            class_id=class_ids
        )

        # Update tracker - adds track_id to detections
        tracked_detections = self.tracker.update_with_detections(detections)

        return tracked_detections

    def reset(self):
        """Reset tracker state (e.g., on camera reconnect)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            # ByteTrack proxy requires positional args
            self.tracker = ByteTrack(
                self.tracker.track_activation_threshold,
                self.tracker.minimum_matching_threshold,
                self.tracker.max_time_lost,
                self.tracker.minimum_consecutive_frames
            )


def test_tracker():
    """Simple test of tracker wrapper with synthetic data."""
    import numpy as np

    wrapper = TrackerWrapper()

    # Simulate a few frames with a moving box
    for frame_idx in range(10):
        # Box moves right each frame
        x1 = 100 + frame_idx * 10
        y1 = 100
        x2 = 200 + frame_idx * 10
        y2 = 200

        boxes = np.array([[x1, y1, x2, y2]], dtype=np.float32)
        confidences = np.array([0.9], dtype=np.float32)
        class_ids = np.array([0], dtype=np.int32)

        tracked = wrapper.update(boxes, confidences, class_ids)
        print(f"Frame {frame_idx}: track_ids={tracked.tracker_id}, xyxy={tracked.xyxy}")

    # Simulate occlusion (no detections)
    print("\n--- Occlusion ---")
    for i in range(5):
        tracked = wrapper.update(np.empty((0, 4)), np.empty(0), np.empty(0, dtype=np.int32))
        print(f"Occlusion frame {i}: track_ids={tracked.tracker_id}")

    # Target reappears
    print("\n--- Reappearance ---")
    boxes = np.array([[150, 100, 250, 200]], dtype=np.float32)
    confidences = np.array([0.85], dtype=np.float32)
    class_ids = np.array([0], dtype=np.int32)
    tracked = wrapper.update(boxes, confidences, class_ids)
    print(f"Reappearance: track_ids={tracked.tracker_id}, xyxy={tracked.xyxy}")


if __name__ == '__main__':
    test_tracker()
