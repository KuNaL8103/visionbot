#!/usr/bin/env python3
"""
Controller Node

Subscribes to /target/pose (visual_teleop_msgs/TrackedTarget), computes
velocity commands using a simple proportional controller, publishes /cmd_vel
for TurtleBot3 (Gazebo expects geometry_msgs/TwistStamped).

Includes watchdog timer to publish zero Twist when target lost for longer than
target_lost_timeout_sec, and latency logging from frame timestamp.
"""

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from visual_teleop_msgs.msg import TrackedTarget
from geometry_msgs.msg import Twist, TwistStamped


# Match perception_node: only the newest target matters for control.
TARGET_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
)


class ControllerNode(Node):
    """Node for converting target pose to velocity commands."""

    # Calibration for d ~= k / h: choose k so h=0.5 maps to target_distance.
    REFERENCE_BBOX_HEIGHT = 0.5

    def __init__(self):
        super().__init__('controller_node')

        # Declare parameters (loaded from config/params.yaml)
        self.declare_parameter('linear_gain', 0.5)
        self.declare_parameter('angular_gain', 1.0)
        self.declare_parameter('max_linear_speed', 0.22)
        self.declare_parameter('max_angular_speed', 1.82)
        self.declare_parameter('target_distance', 1.0)  # desired distance from target (meters)
        self.declare_parameter('deadband', 0.1)  # distance deadband (meters)
        self.declare_parameter('dead_zone_px', 0.05)  # dead zone in normalized x (0-1)
        self.declare_parameter('target_lost_timeout_sec', 1.0)  # seconds - watchdog timeout for zero Twist
        self.declare_parameter('enable_safety_stop', True)
        self.declare_parameter('publish_rate_hz', 60.0)  # Faster cmd_vel dispatch

        # Get parameter values
        self.linear_gain = self.get_parameter('linear_gain').get_parameter_value().double_value
        self.angular_gain = self.get_parameter('angular_gain').get_parameter_value().double_value
        self.max_linear_speed = self.get_parameter('max_linear_speed').get_parameter_value().double_value
        self.max_angular_speed = self.get_parameter('max_angular_speed').get_parameter_value().double_value
        self.target_distance = self.get_parameter('target_distance').get_parameter_value().double_value
        self.deadband = self.get_parameter('deadband').get_parameter_value().double_value
        self.dead_zone_px = self.get_parameter('dead_zone_px').get_parameter_value().double_value
        self.target_lost_timeout_sec = self.get_parameter('target_lost_timeout_sec').get_parameter_value().double_value
        self.enable_safety_stop = self.get_parameter('enable_safety_stop').get_parameter_value().bool_value
        self.publish_rate_hz = self.get_parameter('publish_rate_hz').get_parameter_value().double_value

        self.get_logger().info(f'Controller node initialized')
        self.get_logger().info(f'  linear_gain: {self.linear_gain}')
        self.get_logger().info(f'  angular_gain: {self.angular_gain}')
        self.get_logger().info(f'  max_linear_speed: {self.max_linear_speed}')
        self.get_logger().info(f'  max_angular_speed: {self.max_angular_speed}')
        self.get_logger().info(f'  target_distance: {self.target_distance}')
        self.get_logger().info(f'  deadband: {self.deadband}')
        self.get_logger().info(f'  dead_zone_px: {self.dead_zone_px}')
        self.get_logger().info(f'  target_lost_timeout_sec: {self.target_lost_timeout_sec}')
        self.get_logger().info(f'  enable_safety_stop: {self.enable_safety_stop}')
        self.get_logger().info(f'  publish_rate_hz: {self.publish_rate_hz}')

        # Subscriber to target pose (depth=1 drops stale queued targets).
        self.target_sub = self.create_subscription(
            TrackedTarget,
            '/target/pose',
            self.target_callback,
            TARGET_QOS
        )

        # Publisher for cmd_vel (Gazebo bridge expects TwistStamped)
        self.cmd_vel_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)

        # Store latest target and timestamp
        self.latest_target = None
        self.last_target_time = None
        self.last_latency_sec = None  # Stores last computed frame-to-cmd_vel latency (seconds)

        # Timer to publish cmd_vel at configured rate
        timer_period = 1.0 / self.publish_rate_hz
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Watchdog timer: publishes zero Twist if target lost for longer than target_lost_timeout_sec
        self.watchdog_timer = self.create_timer(0.1, self.watchdog_callback)  # 10 Hz check

    @staticmethod
    def _stamp_ns(stamp) -> int:
        """Convert a builtin_interfaces/Time message to nanoseconds."""
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def target_callback(self, msg: TrackedTarget):
        """Callback for target pose messages, dropping out-of-order samples."""
        new_stamp_ns = self._stamp_ns(msg.stamp)
        if self.latest_target is not None:
            old_stamp_ns = self._stamp_ns(self.latest_target.stamp)
            if new_stamp_ns > 0 and old_stamp_ns > 0 and new_stamp_ns < old_stamp_ns:
                self.get_logger().debug(
                    'Dropping out-of-order target message (older frame stamp)')
                return

        self.latest_target = msg
        self.last_target_time = self.get_clock().now()

    def watchdog_callback(self):
        """Watchdog timer: publishes zero Twist if target lost for longer than target_lost_timeout_sec."""
        if self.latest_target is None:
            # No target ever received - publish zero (safety)
            if self.enable_safety_stop:
                self._publish_zero_twist()
            return

        # Check if target_visible has been false for longer than target_lost_timeout_sec
        if not self.latest_target.target_visible:
            elapsed = (self.get_clock().now() - self.last_target_time).nanoseconds / 1e9
            if elapsed > self.target_lost_timeout_sec:
                if self.enable_safety_stop:
                    self.get_logger().debug(f'Watchdog: target lost for {elapsed:.2f}s (> {self.target_lost_timeout_sec}s), publishing zero Twist')
                    self._publish_zero_twist()

    def _publish_zero_twist(self):
        """Publish zero velocity command."""
        twist_stamped = TwistStamped()
        twist_stamped.header.stamp = self.get_clock().now().to_msg()
        twist_stamped.header.frame_id = 'base_link'
        self.cmd_vel_pub.publish(twist_stamped)

    def timer_callback(self):
        """Periodic callback to compute and publish cmd_vel."""
        twist_stamped = self.compute_cmd_vel()
        self.cmd_vel_pub.publish(twist_stamped)

    def _distance_from_bbox_height(self, bbox_height: float) -> float:
        """Estimate distance from normalized bbox height using d ~= k / h (pinhole heuristic)."""
        return self.target_distance * self.REFERENCE_BBOX_HEIGHT / bbox_height

    def compute_cmd_vel(self) -> TwistStamped:
        """Compute velocity command from latest target."""
        now = self.get_clock().now()
        twist_stamped = TwistStamped()
        twist_stamped.header.stamp = now.to_msg()
        twist_stamped.header.frame_id = 'base_link'
        twist = Twist()

        # Check if we have a recent target
        if self.latest_target is None:
            self.last_latency_sec = None
            if self.enable_safety_stop:
                self.get_logger().debug('No target received yet, publishing zero velocity')
            return twist_stamped

        # Check if target is visible
        if not self.latest_target.target_visible:
            # Never accumulate latency from occluded/coasting frames.
            self.last_latency_sec = None
            if self.enable_safety_stop:
                self.get_logger().debug('Target not visible, publishing zero velocity')
            return twist_stamped

        # Check if target is recent (safety timeout)
        if self.last_target_time is not None:
            elapsed = (now - self.last_target_time).nanoseconds / 1e9
            if elapsed > self.target_lost_timeout_sec:
                self.last_latency_sec = None
                if self.enable_safety_stop:
                    self.get_logger().debug(f'Target timeout ({elapsed:.2f}s), publishing zero velocity')
                return twist_stamped

        # Reject a message that arrived on time but carries a stale frame stamp
        # (e.g. a queued frame from before the capture-thread fix). Stamp 0 is
        # treated as "not provided" for compatibility with synthetic messages.
        frame_stamp_ns = self._stamp_ns(self.latest_target.stamp)
        frame_age_sec = None
        if frame_stamp_ns > 0:
            frame_time = rclpy.time.Time.from_msg(self.latest_target.stamp)
            frame_age_sec = (now - frame_time).nanoseconds / 1e9
            if frame_age_sec < 0.0 or frame_age_sec > self.target_lost_timeout_sec:
                self.last_latency_sec = None
                self.get_logger().warn(
                    f'Ignoring stale frame stamp ({frame_age_sec * 1000:.1f}ms old)',
                    throttle_duration_sec=2.0,
                )
                return twist_stamped

        # Extract target position (normalized 0-1)
        target_x = self.latest_target.x
        target_y = self.latest_target.y

        # Compute horizontal error from image center (0.5)
        error_x = target_x - 0.5

        # Angular velocity: proportional to horizontal error
        # error_x = target_x - 0.5: negative when target is left, positive when right
        # ROS convention: angular.z > 0 = CCW (turn LEFT), angular.z < 0 = CW (turn RIGHT)
        # Target on LEFT (x < 0.5) -> robot must turn LEFT -> angular.z > 0
        # Target on RIGHT (x > 0.5) -> robot must turn RIGHT -> angular.z < 0
        # Therefore: angular_vel = -error_x * angular_gain
        if abs(error_x) < self.dead_zone_px:
            angular_vel = 0.0
        else:
            angular_vel = -error_x * self.angular_gain
            # Clamp to max angular speed
            angular_vel = max(-self.max_angular_speed, min(self.max_angular_speed, angular_vel))

        # Linear velocity: hold target_distance when bbox height is available.
        # Older/synthetic messages without a height retain the legacy behavior.
        bbox_height = float(getattr(self.latest_target, 'bbox_height', 0.0))
        if abs(error_x) < 0.3 and bbox_height > 1e-3:
            estimated_distance = self._distance_from_bbox_height(bbox_height)
            distance_error = estimated_distance - self.target_distance
            if abs(distance_error) <= self.deadband:
                linear_vel = 0.0
            else:
                linear_vel = self.linear_gain * distance_error
                linear_vel = max(
                    -self.max_linear_speed,
                    min(self.max_linear_speed, linear_vel),
                )
        elif abs(error_x) < 0.3:
            linear_vel = self.linear_gain
            linear_vel = max(0.0, min(self.max_linear_speed, linear_vel))
        else:
            linear_vel = 0.0

        twist.linear.x = linear_vel
        twist.angular.z = angular_vel

        twist_stamped.twist = twist

        # Latency logging: time from frame capture (msg.stamp) to Twist publish.
        # Only visible, fresh frames reach this point; occluded/coasting frames
        # return early and never contribute a fake 3000ms sample.
        if frame_age_sec is not None:
            self.last_latency_sec = frame_age_sec
            self.get_logger().info(
                f'Latency: {frame_age_sec * 1000:.1f}ms (frame->cmd_vel)',
                throttle_duration_sec=2.0,
            )
        else:
            self.last_latency_sec = None

        return twist_stamped

    def destroy_node(self):
        # Publish zero velocity on shutdown, but never crash if the ROS context
        # was already shut down by Ctrl+C / the signal handler.
        if rclpy.ok():
            try:
                zero_twist = TwistStamped()
                zero_twist.header.stamp = self.get_clock().now().to_msg()
                zero_twist.header.frame_id = 'base_link'
                self.cmd_vel_pub.publish(zero_twist)
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == '__main__':
    main()