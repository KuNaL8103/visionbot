#!/usr/bin/env python3
"""Regression tests for the low-latency capture, QoS, and latency guards."""

import threading
import unittest

import numpy as np
import rclpy

from visual_teleop.controller_node import ControllerNode
from visual_teleop.perception_node import PerceptionNode
from visual_teleop_msgs.msg import TrackedTarget


class TestLatestFrameSlot(unittest.TestCase):
    """The capture slot must expose a frame once and skip duplicates."""

    def setUp(self):
        # Bare object: no ROS node, camera, or background thread needed.
        self.node = object.__new__(PerceptionNode)
        self.node._frame_lock = threading.Lock()
        self.node._latest_frame = None
        self.node._latest_frame_stamp = None
        self.node._frame_seq = 0
        self.node._processed_frame_seq = 0

    def test_new_frame_is_returned_once_then_skipped(self):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        self.node._latest_frame = frame
        self.node._frame_seq = 1

        ret, got_frame, _ = self.node._take_latest_frame()
        self.assertTrue(ret)
        self.assertIs(got_frame, frame)

        ret, got_frame, _ = self.node._take_latest_frame()
        self.assertFalse(ret)
        self.assertIsNone(got_frame)

    def test_newer_frame_is_returned_after_skipping(self):
        self.node._latest_frame = np.zeros((4, 4, 3), dtype=np.uint8)
        self.node._frame_seq = 1
        self.node._take_latest_frame()

        newer = np.ones((4, 4, 3), dtype=np.uint8)
        self.node._latest_frame = newer
        self.node._frame_seq = 2

        ret, got_frame, _ = self.node._take_latest_frame()
        self.assertTrue(ret)
        self.assertIs(got_frame, newer)

    def test_auto_device_resolves_to_a_real_backend(self):
        device, cuda_available = PerceptionNode._resolve_device('auto')
        try:
            import torch
            expected = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        except Exception:
            expected = 'cpu'

        self.assertEqual(device, expected)
        self.assertEqual(cuda_available, expected == 'cuda:0')


class TestControllerLatencyGuards(unittest.TestCase):
    """Stale/out-of-order targets must not produce 3000ms latency or motion."""

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        if rclpy.ok():
            rclpy.shutdown()

    def setUp(self):
        self.node = ControllerNode()

    def tearDown(self):
        self.node.destroy_node()

    @staticmethod
    def _target(x, stamp=None):
        msg = TrackedTarget()
        msg.x = x
        msg.y = 0.5
        msg.confidence = 0.9
        msg.target_visible = True
        msg.track_id = 1
        if stamp is not None:
            msg.stamp = stamp
        return msg

    def test_stale_frame_stamp_produces_zero_and_no_latency(self):
        now = self.node.get_clock().now()
        stale_stamp = rclpy.time.Time(
            nanoseconds=now.nanoseconds - 3_000_000_000
        ).to_msg()

        self.node.target_callback(self._target(0.5, stale_stamp))
        twist = self.node.compute_cmd_vel()

        self.assertEqual(twist.twist.linear.x, 0.0)
        self.assertEqual(twist.twist.angular.z, 0.0)
        self.assertIsNone(self.node.last_latency_sec)

    def test_out_of_order_target_is_dropped(self):
        now = self.node.get_clock().now()
        newer_stamp = rclpy.time.Time(nanoseconds=now.nanoseconds).to_msg()
        older_stamp = rclpy.time.Time(
            nanoseconds=now.nanoseconds - 100_000_000
        ).to_msg()

        self.node.target_callback(self._target(0.7, newer_stamp))
        self.node.target_callback(self._target(0.1, older_stamp))

        self.assertAlmostEqual(self.node.latest_target.x, 0.7, places=6)

    def test_occluded_frame_does_not_accumulate_latency(self):
        now = self.node.get_clock().now()
        stamp = rclpy.time.Time(nanoseconds=now.nanoseconds).to_msg()
        msg = self._target(0.5, stamp)
        msg.target_visible = False

        self.node.target_callback(msg)
        twist = self.node.compute_cmd_vel()

        self.assertEqual(twist.twist.linear.x, 0.0)
        self.assertIsNone(self.node.last_latency_sec)


if __name__ == '__main__':
    unittest.main()
