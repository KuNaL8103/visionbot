# Visual Teleoperation Topic Contracts

## Topic Summary

| Topic | Type | Publisher | Subscriber | Rate |
|-------|------|-----------|------------|------|
| `/target/pose` | visual_teleop_msgs/TrackedTarget | perception_node | controller_node | ~30 Hz |
| `/perception/annotated` | sensor_msgs/Image | perception_node | RViz / debug tools | ~30 Hz |
| `/cmd_vel` | geometry_msgs/TwistStamped | controller_node | TurtleBot3 (Gazebo) | 60 Hz |

---

## `/target/pose`

**Publisher**: perception_node
**Subscriber**: controller_node
**QoS**: Best effort, keep last 1 (latest target only)

```python
# visual_teleop_msgs.msg.TrackedTarget
x: float32                          # Normalized X in camera frame (0.0..1.0, origin top-left)
y: float32                          # Normalized Y in camera frame (0.0..1.0, origin top-left)
bbox_height: float32                # Normalized bbox height; inverse-distance proxy
confidence: float32                 # Detection confidence 0.0..1.0
target_visible: bool                # True if target currently tracked
track_id: int32                     # ByteTrack track ID (persistent across frames)
stamp: builtin_interfaces/Time      # Frame capture timestamp for latency measurement
```

**Coordinate Frame**: Camera image frame (origin top-left, X right, Y down)
- `x`, `y`: Normalized coordinates (0.0..1.0) of tracked target center
  - `x = 0.0` = left edge, `x = 1.0` = right edge
  - `y = 0.0` = top edge, `y = 1.0` = bottom edge
  - Image center is at `(0.5, 0.5)`
- `bbox_height`: normalized bbox height; the controller uses `d ~= 0.5 * target_distance / bbox_height` as a pinhole distance proxy
- `confidence`: YOLO detection confidence for this track
- `target_visible`: False when target lost (track expired)
- `track_id`: ByteTrack-assigned persistent track ID
- `stamp`: Timestamp of frame capture (used by controller for latency logging)

**Special Values**:
- If no target detected: publish with `target_visible: false`, `track_id: 0`, `confidence: 0.0`, holding last known `x`, `y`
- If multiple targets: publish best track (highest confidence)

---

## `/perception/annotated`

**Publisher**: perception_node
**Subscriber**: RViz / debug tools
**QoS**: Sensor data preset (best effort, keep last 5)

`sensor_msgs/Image` (`bgr8`) with the tracked bbox, track ID, confidence, FPS,
inference time, and smoothing mode drawn on the frame.

---

## `/cmd_vel`

**Publisher**: controller_node
**Subscriber**: TurtleBot3 (Gazebo) / diff_drive_controller (via ros_gz_bridge)
**QoS**: Reliable, keep last 10

```python
# geometry_msgs.msg.TwistStamped
header:
  stamp: builtin_interfaces/Time    # Current time
  frame_id: string                  # "base_link"
twist:
  linear:
    x: float64                      # Forward velocity (m/s)
    y: 0.0                          # Unused (differential drive)
    z: 0.0
  angular:
    x: 0.0
    y: 0.0
    z: float64                      # Yaw rate (rad/s)
```

**Limits** (TurtleBot3 Burger):
- `linear.x`: [-0.22, 0.22] m/s
- `angular.z`: [-1.82, 1.82] rad/s

**Behavior**:
- Positive `linear.x`: forward; negative `linear.x`: back away
- `target_distance` / `deadband` hold distance using normalized bbox height
- Positive `angular.z`: counter-clockwise (left turn)
- Zero velocities when `target_visible=false` for > `target_lost_timeout_sec` (default 1.0s)
- Watchdog timer (10 Hz) ensures zero TwistStamped published even if no new `/target/pose` messages arrive

---

## Message Contract Rules

1. **Do not change topic names** without updating both ARCHITECTURE.md and this file
2. **Do not change message types** without updating all publishers/subscribers
3. **Timestamp synchronization**: perception_node uses frame acquisition time (`stamp`), not processing time
4. **QoS**: `/target/pose` uses best-effort depth 1 to drop stale targets; `/cmd_vel` remains reliable depth 10