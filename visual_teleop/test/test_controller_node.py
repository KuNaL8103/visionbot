#!/usr/bin/env python3
"""
Test for Controller Node

Uses unittest with mocked TrackedTarget inputs to test controller logic
without requiring a camera or simulation.
"""

import unittest

import rclpy
from geometry_msgs.msg import TwistStamped

from visual_teleop.controller_node import ControllerNode
from visual_teleop_msgs.msg import TrackedTarget


class TestControllerNode(unittest.TestCase):
    """Test cases for controller_node logic."""

    @classmethod
    def setUpClass(cls):
        """Initialize ROS 2 context once for all tests."""
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        """Shutdown ROS 2 context."""
        rclpy.shutdown()

    def setUp(self):
        """Create a fresh controller node for each test."""
        self.node = ControllerNode()

    def tearDown(self):
        """Clean up node."""
        self.node.destroy_node()

    def _create_target(self, x: float, y: float, target_visible: bool = True,
                       confidence: float = 0.8, track_id: int = 1,
                       bbox_height: float = 0.0) -> TrackedTarget:
        """Helper to create a TrackedTarget message."""
        msg = TrackedTarget()
        msg.x = x
        msg.y = y
        msg.bbox_height = bbox_height
        msg.confidence = confidence
        msg.target_visible = target_visible
        msg.track_id = track_id
        return msg

    def test_dead_center_produces_near_zero_angular(self):
        """Target at image center (x=0.5) should produce near-zero angular velocity."""
        target = self._create_target(x=0.5, y=0.5)
        self.node.target_callback(target)
        twist_stamped = self.node.compute_cmd_vel()

        # Angular velocity should be zero (within dead_zone_px)
        self.assertAlmostEqual(
            twist_stamped.twist.angular.z, 0.0, places=5,
            msg='angular.z should be near zero')

        # Linear velocity should be positive (target is centered)
        self.assertGreater(twist_stamped.twist.linear.x, 0.0,
                           msg=f"Expected positive linear.x, got {twist_stamped.twist.linear.x}")
        self.assertLessEqual(twist_stamped.twist.linear.x, self.node.max_linear_speed)

    def test_far_left_produces_positive_angular(self):
        """Target far left (x=0.1) should produce positive angular velocity (turn LEFT/CCW)."""
        target = self._create_target(x=0.1, y=0.5)
        self.node.target_callback(target)
        twist_stamped = self.node.compute_cmd_vel()

        # Angular velocity should be positive (turn LEFT/CCW to center target on left)
        # ROS convention: angular.z > 0 = CCW = turn LEFT
        # Target on LEFT side of image (x < 0.5) -> robot must turn LEFT -> angular.z > 0
        self.assertGreater(
            twist_stamped.twist.angular.z, 0.0,
            msg='left target should produce positive angular.z')
        # Magnitude should be clamped to max_angular_speed
        self.assertLessEqual(twist_stamped.twist.angular.z, self.node.max_angular_speed)

        # Linear velocity should be zero (target too far left, not forward-facing)
        self.assertEqual(
            twist_stamped.twist.linear.x, 0.0,
            msg='far-left target should not move forward')

    def test_far_right_produces_negative_angular(self):
        """Target far right (x=0.9) should produce negative angular velocity (turn RIGHT/CW)."""
        target = self._create_target(x=0.9, y=0.5)
        self.node.target_callback(target)
        twist_stamped = self.node.compute_cmd_vel()

        # Angular velocity should be negative (turn RIGHT/CW to center target on right)
        # ROS convention: angular.z < 0 = CW = turn RIGHT
        # Target on RIGHT side of image (x > 0.5) -> robot must turn RIGHT -> angular.z < 0
        self.assertLess(
            twist_stamped.twist.angular.z, 0.0,
            msg='right target should produce negative angular.z')
        # Magnitude should be clamped to max_angular_speed
        self.assertGreaterEqual(twist_stamped.twist.angular.z, -self.node.max_angular_speed)

        # Linear velocity should be zero (target too far right, not forward-facing)
        self.assertEqual(
            twist_stamped.twist.linear.x, 0.0,
            msg='far-right target should not move forward')

    def test_target_not_visible_produces_zero_twist(self):
        """target_visible=False should produce zero Twist (stop)."""
        target = self._create_target(x=0.5, y=0.5, target_visible=False)
        self.node.target_callback(target)
        twist_stamped = self.node.compute_cmd_vel()

        # Both linear and angular should be zero
        self.assertEqual(
            twist_stamped.twist.linear.x, 0.0,
            msg='linear.x should be zero when target is not visible')
        self.assertEqual(
            twist_stamped.twist.angular.z, 0.0,
            msg='angular.z should be zero when target is not visible')

    def test_no_target_received_produces_zero_twist(self):
        """No target received yet should produce zero Twist."""
        # Don't call target_callback at all
        twist_stamped = self.node.compute_cmd_vel()

        self.assertEqual(
            twist_stamped.twist.linear.x, 0.0,
            msg='linear.x should be zero with no target')
        self.assertEqual(
            twist_stamped.twist.angular.z, 0.0,
            msg='angular.z should be zero with no target')

    def test_target_within_dead_zone_produces_zero_angular(self):
        """Target within dead_zone_px of center should produce zero angular velocity."""
        # dead_zone_px default is 0.05, so target at x=0.52 is within dead zone
        target = self._create_target(x=0.52, y=0.5)
        self.node.target_callback(target)
        twist_stamped = self.node.compute_cmd_vel()

        self.assertAlmostEqual(
            twist_stamped.twist.angular.z, 0.0, places=5,
            msg='angular.z should be zero inside dead zone')

    def test_angular_velocity_clamped_to_max(self):
        """Angular velocity should be clamped to max_angular_speed."""
        # Use very large error to exceed max
        target = self._create_target(x=0.0, y=0.5)  # max error = -0.5
        self.node.target_callback(target)
        twist_stamped = self.node.compute_cmd_vel()

        # With angular_gain=1.0, error=-0.5 -> angular_vel=0.5 (within max 1.82)
        # But with larger gain it would clamp. Let's verify it doesn't exceed max.
        self.assertLessEqual(
            abs(twist_stamped.twist.angular.z),
            self.node.max_angular_speed + 1e-6,
            msg='angular velocity exceeds max')

    def test_linear_velocity_clamped_to_max(self):
        """Linear velocity should be clamped to max_linear_speed."""
        # Target centered with high confidence
        target = self._create_target(x=0.5, y=0.5, confidence=1.0)
        self.node.target_callback(target)
        twist_stamped = self.node.compute_cmd_vel()

        self.assertLessEqual(
            twist_stamped.twist.linear.x,
            self.node.max_linear_speed + 1e-6,
            msg='linear velocity exceeds max')
        self.assertGreaterEqual(twist_stamped.twist.linear.x, 0.0)

    def test_target_slightly_left_still_moves_forward(self):
        """Target slightly left (within forward-facing range) should still move forward."""
        # forward-facing range is abs(error_x) < 0.3, so x=0.3 is at boundary
        target = self._create_target(x=0.3, y=0.5)
        self.node.target_callback(target)
        twist_stamped = self.node.compute_cmd_vel()

        # Should have positive linear velocity (moving forward)
        self.assertGreater(twist_stamped.twist.linear.x, 0.0)
        # Should have positive angular velocity (turning LEFT/CCW toward target on left)
        self.assertGreater(twist_stamped.twist.angular.z, 0.0)

    def test_distance_hold_backs_up_when_target_is_close(self):
        """A large bbox height means the target is too close, so linear.x < 0."""
        target = self._create_target(x=0.5, y=0.5, bbox_height=0.75)
        self.node.target_callback(target)
        twist_stamped = self.node.compute_cmd_vel()

        self.assertLess(twist_stamped.twist.linear.x, 0.0)
        self.assertGreaterEqual(twist_stamped.twist.linear.x, -self.node.max_linear_speed)

    def test_distance_hold_moves_forward_when_target_is_far(self):
        """A small bbox height means the target is too far, so linear.x > 0."""
        target = self._create_target(x=0.5, y=0.5, bbox_height=0.25)
        self.node.target_callback(target)
        twist_stamped = self.node.compute_cmd_vel()

        self.assertGreater(twist_stamped.twist.linear.x, 0.0)
        self.assertLessEqual(twist_stamped.twist.linear.x, self.node.max_linear_speed)

    def test_distance_hold_inside_deadband_stops(self):
        """Distance error inside deadband should produce zero linear velocity."""
        target = self._create_target(x=0.5, y=0.5, bbox_height=0.55)
        self.node.target_callback(target)
        twist_stamped = self.node.compute_cmd_vel()

        self.assertEqual(twist_stamped.twist.linear.x, 0.0)

    def test_target_slightly_right_still_moves_forward(self):
        """Target slightly right (within forward-facing range) should still move forward."""
        target = self._create_target(x=0.7, y=0.5)
        self.node.target_callback(target)
        twist_stamped = self.node.compute_cmd_vel()

        # Should have positive linear velocity (moving forward)
        self.assertGreater(twist_stamped.twist.linear.x, 0.0)
        # Should have negative angular velocity (turning RIGHT/CW toward target on right)
        self.assertLess(twist_stamped.twist.angular.z, 0.0)

    def test_watchdog_publishes_zero_twist_when_target_lost_no_new_messages(self):
        """Watchdog should stop the robot after a target-lost timeout."""
        # First, send a visible target
        target = self._create_target(x=0.5, y=0.5, target_visible=True)
        self.node.target_callback(target)

        # Capture published messages from watchdog
        published_messages = []

        def capture_publish(msg):
            published_messages.append(msg)
        original_publish = self.node.cmd_vel_pub.publish
        self.node.cmd_vel_pub.publish = capture_publish

        # Now send target_visible=False
        target = self._create_target(x=0.5, y=0.5, target_visible=False)
        self.node.target_callback(target)

        # Wait a bit to simulate time passing (we can't easily mock time in ROS,
        # so we'll directly manipulate the last_target_time to simulate elapsed time)
        # The watchdog checks: elapsed = (now - last_target_time) > target_lost_timeout_sec
        # We'll set last_target_time to be older than target_lost_timeout_sec
        old_time = self.node.get_clock().now() - rclpy.duration.Duration(
            seconds=self.node.target_lost_timeout_sec + 0.5
        )
        self.node.last_target_time = old_time

        # Call watchdog_callback directly (simulates 10Hz timer firing)
        self.node.watchdog_callback()

        # Verify zero Twist was published by watchdog
        self.assertEqual(
            len(published_messages), 1,
            msg='watchdog should publish exactly one zero Twist')
        msg = published_messages[0]
        self.assertEqual(msg.twist.linear.x, 0.0, "Watchdog should publish zero linear.x")
        self.assertEqual(msg.twist.angular.z, 0.0, "Watchdog should publish zero angular.z")

        # Restore
        self.node.cmd_vel_pub.publish = original_publish

    def test_latency_logging_with_valid_stamp(self):
        """Latency should be sane for a fresh frame timestamp."""
        # Create a target with stamp from 50ms ago
        target = self._create_target(x=0.5, y=0.5, target_visible=True)

        # Set stamp to 50ms in the past using total nanoseconds to avoid underflow
        now = self.node.get_clock().now()
        past_time = rclpy.time.Time(nanoseconds=now.nanoseconds - 50_000_000)
        target.stamp = past_time.to_msg()

        self.node.target_callback(target)
        twist_stamped = self.node.compute_cmd_vel()

        # The function should not crash (no exception)
        self.assertIsNotNone(twist_stamped)
        self.assertIsInstance(twist_stamped, TwistStamped)

        # Verify twist values are reasonable for centered target
        self.assertAlmostEqual(twist_stamped.twist.angular.z, 0.0, places=5)
        self.assertGreater(twist_stamped.twist.linear.x, 0.0)
        self.assertLessEqual(twist_stamped.twist.linear.x, self.node.max_linear_speed)

        # Verify computed latency is exposed and close to expected 50ms (0.05s)
        # Allow some tolerance for execution time (±20ms)
        self.assertIsNotNone(self.node.last_latency_sec, "last_latency_sec should be set")
        self.assertGreater(self.node.last_latency_sec, 0.02,
                           "Latency should be positive and > 20ms")
        self.assertLess(self.node.last_latency_sec, 0.1,
                        "Latency should be < 100ms (close to 50ms)")


if __name__ == '__main__':
    unittest.main()
