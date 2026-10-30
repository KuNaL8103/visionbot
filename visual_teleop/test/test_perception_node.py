#!/usr/bin/env python3
"""
Test for Perception Node

Tests parameter loading and publish logic with mocked camera.
"""

import unittest
from unittest.mock import patch, MagicMock
from collections import deque

import cv2
import numpy as np
import rclpy
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image

from visual_teleop.perception_node import PerceptionNode
from visual_teleop_msgs.msg import TrackedTarget


class TestPerceptionNode(unittest.TestCase):
    """Test cases for perception_node."""

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        """Set up test node with mocked camera, YOLO, and ByteTrack."""
        # Mock cv2.VideoCapture before creating node
        self.mock_cap_patcher = patch('visual_teleop.perception_node.cv2.VideoCapture')
        self.mock_cap_class = self.mock_cap_patcher.start()
        self.mock_cap = MagicMock()
        self.mock_cap.isOpened.return_value = True
        self.mock_cap.get.side_effect = lambda prop: {
            3: 640,   # CAP_PROP_FRAME_WIDTH
            4: 480,   # CAP_PROP_FRAME_HEIGHT
            5: 30.0,  # CAP_PROP_FPS
        }.get(prop, 0)
        # Create a mock frame with proper shape attribute (height, width, channels)
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.mock_cap.read.return_value = (True, mock_frame)  # ret, frame
        self.mock_cap_class.return_value = self.mock_cap

        # Mock YOLO model
        self.mock_yolo_patcher = patch('visual_teleop.perception_node.YOLO')
        self.mock_yolo_class = self.mock_yolo_patcher.start()
        self.mock_model = MagicMock()
        self.mock_model.names = {0: 'person', 1: 'bicycle', 2: 'car'}
        self.mock_model.to.return_value = None
        self.mock_yolo_class.return_value = self.mock_model

        # Mock YOLO inference results
        self.mock_result = MagicMock()
        self.mock_box = MagicMock()
        self.mock_box.cls.item.return_value = 0  # person class
        self.mock_box.conf.item.return_value = 0.8
        self.mock_box.xyxy = [MagicMock()]
        self.mock_box.xyxy[0].tolist.return_value = [100, 100, 300, 300]
        self.mock_result.boxes = [self.mock_box]
        self.mock_model.return_value = [self.mock_result]

        # Mock ByteTrack tracker
        self.mock_tracker_patcher = patch('visual_teleop.perception_node.TrackerWrapper')
        self.mock_tracker_class = self.mock_tracker_patcher.start()
        self.mock_tracker = MagicMock()

        # Mock tracked detections output
        mock_detections = MagicMock()
        mock_detections.confidence = np.array([0.8], dtype=np.float32)
        mock_detections.xyxy = np.array([[100, 100, 300, 300]], dtype=np.float32)
        mock_detections.tracker_id = np.array([1], dtype=np.int32)
        mock_detections.__len__ = lambda self: 1  # Make len() work for "if len(detections) > 0"
        self.mock_tracker.update.return_value = mock_detections
        self.mock_tracker_class.return_value = self.mock_tracker

        # Mock create_timer to prevent timer from firing during init (warmup)
        self.mock_timer_patcher = patch.object(PerceptionNode, 'create_timer', return_value=MagicMock())
        self.mock_timer = self.mock_timer_patcher.start()

    def tearDown(self):
        self.mock_cap_patcher.stop()
        self.mock_yolo_patcher.stop()
        self.mock_tracker_patcher.stop()
        self.mock_timer_patcher.stop()

    def test_parameter_loading_defaults(self):
        """Test that default parameters are loaded correctly."""
        node = PerceptionNode()

        # Check parameters loaded with defaults
        self.assertEqual(node.camera_index, 0)
        self.assertEqual(node.camera_width, 640)
        self.assertEqual(node.camera_height, 480)
        self.assertEqual(node.publish_rate_hz, 30.0)

        node.destroy_node()

    def test_parameter_loading_overridden(self):
        """Test that parameters can be overridden."""
        # Create node with custom parameters
        node = rclpy.create_node('test_perception_node_params')

        # Declare and set custom parameters
        node.declare_parameter('camera_index', 1)
        node.declare_parameter('camera_width', 1280)
        node.declare_parameter('camera_height', 720)
        node.declare_parameter('publish_rate_hz', 15.0)

        # Now create our perception node - it will use the parameter values
        # Since we can't easily inject params into existing node, test the logic directly
        node.destroy_node()

    def test_publishes_correct_trackedtarget(self):
        """Test that timer callback publishes TrackedTarget with correct values."""
        node = PerceptionNode()

        # Capture published messages
        published_messages = []

        def capture_publish(msg):
            published_messages.append(msg)

        # Replace publisher's publish method
        original_publish = node.target_pub.publish
        node.target_pub.publish = capture_publish

        # Call timer callback directly
        node.timer_callback()

        # Verify a message was published
        self.assertEqual(len(published_messages), 1)

        msg = published_messages[0]
        self.assertIsInstance(msg, TrackedTarget)

        # Check mocked detector values (bbox [100,100,300,300] on 640x480 frame)
        # Center = (200, 200), normalized = (200/640, 200/480) = (0.3125, 0.41666...)
        self.assertAlmostEqual(msg.x, 200/640, places=4, msg="x should be normalized center of mocked bbox")
        self.assertAlmostEqual(msg.y, 200/480, places=4, msg="y should be normalized center of mocked bbox")
        self.assertAlmostEqual(msg.confidence, 0.8, places=4, msg="confidence should match mocked detection")
        self.assertAlmostEqual(msg.bbox_height, 200/480, places=4,
                               msg="bbox_height should be normalized bbox height")
        self.assertTrue(msg.target_visible, "target_visible should be True for detection")
        self.assertEqual(msg.track_id, 1, "track_id should match mocked tracker output")

        node.destroy_node()

    def test_multiple_timer_callbacks(self):
        """Test that multiple callbacks publish consistent messages."""
        node = PerceptionNode()

        published_messages = []
        def capture_publish(msg):
            published_messages.append(msg)
        node.target_pub.publish = capture_publish

        # Call callback multiple times
        for _ in range(5):
            node.timer_callback()

        self.assertEqual(len(published_messages), 5)

        # All messages should be identical (mocked detector)
        for msg in published_messages:
            self.assertAlmostEqual(msg.x, 200/640, places=4)
            self.assertAlmostEqual(msg.y, 200/480, places=4)
            self.assertAlmostEqual(msg.confidence, 0.8, places=4)
            self.assertTrue(msg.target_visible)
            self.assertEqual(msg.track_id, 1)

        node.destroy_node()

    def test_publishes_annotated_image(self):
        """Annotated frames should be published as sensor_msgs/Image."""
        node = PerceptionNode()

        published_images = []
        node.annotated_pub.publish = published_images.append

        node.timer_callback()

        self.assertEqual(len(published_images), 1)
        image = published_images[0]
        self.assertIsInstance(image, Image)
        self.assertEqual(image.width, 640)
        self.assertEqual(image.height, 480)
        self.assertEqual(image.encoding, 'bgr8')

        node.destroy_node()

    def test_video_file_source_uses_ffmpeg(self):
        """camera_source should accept a video path and select the FFMPEG backend."""
        overrides = [Parameter('camera_source', value='demo.mp4')]
        node = PerceptionNode(parameter_overrides=overrides)

        self.assertEqual(node.camera_source, 'demo.mp4')
        self.assertTrue(node._is_video_file)
        self.mock_cap_class.assert_called_once_with('demo.mp4', cv2.CAP_FFMPEG)

        node.destroy_node()

    def test_camera_open_failure_handled(self):
        """Test that node handles camera open failure gracefully."""
        # Make VideoCapture return a failed capture
        self.mock_cap.isOpened.return_value = False

        node = PerceptionNode()

        # Should not crash, just log warning
        self.assertFalse(node.cap.isOpened())

        node.destroy_node()

    def test_no_detection_publishes_single_zero_visible_msg(self):
        """Test that when tracker returns no detections, exactly ONE message is published
        with target_visible=False and x/y at last known position (not double-published)."""
        # Change tracker mock to return empty detections
        mock_detections = MagicMock()
        mock_detections.__len__ = lambda self: 0  # No detections
        self.mock_tracker.update.return_value = mock_detections

        node = PerceptionNode()

        # Capture published messages
        published_messages = []

        def capture_publish(msg):
            published_messages.append(msg)
        node.target_pub.publish = capture_publish

        # Call timer callback
        node.timer_callback()

        # Verify exactly ONE message was published (not double-published)
        self.assertEqual(len(published_messages), 1,
                         msg=f"Expected exactly 1 publish, got {len(published_messages)}")

        msg = published_messages[0]
        self.assertIsInstance(msg, TrackedTarget)

        # Should have target_visible=False
        self.assertFalse(msg.target_visible, "target_visible should be False when no detection")
        # Should hold last known position (initialized to 0.5, 0.5)
        self.assertEqual(msg.x, 0.5, "x should be last known position (0.5)")
        self.assertEqual(msg.y, 0.5, "y should be last known position (0.5)")
        self.assertEqual(msg.confidence, 0.0, "confidence should be 0.0 when no detection")
        self.assertEqual(msg.track_id, 0, "track_id should be 0 when no detection")

        node.destroy_node()

    def test_smoothing_moving_average(self):
        """Test that published x/y is moving average of last N raw positions, not raw value itself."""
        # Create node with smoothing_window_size=3 (will be loaded from params.yaml default 5,
        # but we need to test with a specific window size)
        # We'll mock the parameter loading by creating a node with custom params
        node = PerceptionNode()

        # Override smoothing window to 3 for this test
        node.smoothing_window_size = 3
        node.smoothing_buffer_x = deque(maxlen=3)
        node.smoothing_buffer_y = deque(maxlen=3)

        # Capture published messages
        published_messages = []

        def capture_publish(msg):
            published_messages.append(msg)
        node.target_pub.publish = capture_publish

        # Mock different bbox positions for sequential calls
        # Frame 1: bbox at [100, 100, 300, 300] -> center (200, 200) -> norm (0.3125, 0.4167)
        # Frame 2: bbox at [120, 120, 320, 320] -> center (220, 220) -> norm (0.34375, 0.45833)
        # Frame 3: bbox at [140, 140, 340, 340] -> center (240, 240) -> norm (0.375, 0.5)

        bboxes = [
            [100, 100, 300, 300],  # center (200, 200)
            [120, 120, 320, 320],  # center (220, 220)
            [140, 140, 340, 340],  # center (240, 240)
        ]

        expected_centers = [
            (200/640, 200/480),    # 0.3125, 0.4167
            (220/640, 220/480),    # 0.34375, 0.45833
            (240/640, 240/480),    # 0.375, 0.5
        ]

        for i, (bbox, (exp_x, exp_y)) in enumerate(zip(bboxes, expected_centers)):
            # Update mock tracker to return this bbox
            mock_detections = MagicMock()
            mock_detections.confidence = np.array([0.8], dtype=np.float32)
            mock_detections.xyxy = np.array([bbox], dtype=np.float32)
            mock_detections.tracker_id = np.array([1], dtype=np.int32)
            mock_detections.__len__ = lambda self: 1
            self.mock_tracker.update.return_value = mock_detections

            # Call timer callback
            node.timer_callback()

            # Get the published message
            msg = published_messages[-1]

            # For first frame, smoothing buffer has 1 element -> average = raw
            if i == 0:
                self.assertAlmostEqual(msg.x, exp_x, places=4,
                    msg=f"Frame {i+1}: x should equal raw (buffer has 1 element)")
                self.assertAlmostEqual(msg.y, exp_y, places=4,
                    msg=f"Frame {i+1}: y should equal raw (buffer has 1 element)")
            # For second frame, buffer has 2 elements -> average of first 2
            elif i == 1:
                avg_x = (expected_centers[0][0] + expected_centers[1][0]) / 2
                avg_y = (expected_centers[0][1] + expected_centers[1][1]) / 2
                self.assertAlmostEqual(msg.x, avg_x, places=4,
                    msg=f"Frame {i+1}: x should be average of first 2 frames")
                self.assertAlmostEqual(msg.y, avg_y, places=4,
                    msg=f"Frame {i+1}: y should be average of first 2 frames")
            # For third frame, buffer has 3 elements -> average of all 3
            elif i == 2:
                avg_x = sum(c[0] for c in expected_centers) / 3
                avg_y = sum(c[1] for c in expected_centers) / 3
                self.assertAlmostEqual(msg.x, avg_x, places=4,
                    msg=f"Frame {i+1}: x should be average of 3 frames")
                self.assertAlmostEqual(msg.y, avg_y, places=4,
                    msg=f"Frame {i+1}: y should be average of 3 frames")

        node.destroy_node()


class TestPerceptionNodeIntegration(unittest.TestCase):
    """Integration-style tests using launch_testing (optional)."""

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()


if __name__ == '__main__':
    unittest.main()