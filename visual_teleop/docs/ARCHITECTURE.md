# Visual Teleoperation Architecture

## Overview
This package implements a visual teleoperation system where a laptop webcam tracks a target (hand, object, or face) and drives a simulated TurtleBot3 in Gazebo to follow it.

## Node Graph

```
┌──────────────────┐     /target/pose      ┌──────────────────┐
│  perception_node │ ────────────────────▶ │  controller_node │
│  (OpenCV + YOLO) │  (TrackedTarget)      │  (P Controller)  │
└──────────────────┘                        └────────┬─────────┘
                                                     │
                                                     │ /cmd_vel
                                                     │ (geometry_msgs/TwistStamped)
                                                     ▼
                                            ┌──────────────────┐
                                            │  TurtleBot3      │
                                            │  (Gazebo Sim)    │
                                            └──────────────────┘
```

- `perception_node` opens the webcam directly via `cv2.VideoCapture(camera_index)` — no separate camera driver node or `/camera/image_raw` topic.
- `/target/pose` uses `visual_teleop_msgs/TrackedTarget` (2D normalized image coordinates + metadata + timestamp), NOT `geometry_msgs/PoseStamped`.

## Node Descriptions

### perception_node
- **Input**: Webcam index or video file (`camera_index` / `camera_source`)
- **Processing**:
  1. OpenCV VideoCapture (`CAP_PROP_BUFFERSIZE=1`) is drained by a daemon latest-frame thread; the ROS timer processes at most one new frame per tick. Video files are played back at `camera_fps`
  2. YOLOv8 detects objects in frame (`imgsz=320`, FP16 on CUDA)
  3. ByteTrack associates detections across frames
  4. Filter tracks by `target_class` parameter
  5. Select best track (highest confidence)
  6. Apply a constant-velocity 2D Kalman filter to x/y (default; `smoothing_method: "kalman"`) using a 4-state `[x, y, vx, vy]` model. During brief occlusions, predict-only steps coast the estimate for up to `kalman_coast_frames` (default 15) before freezing. A 5-frame moving average remains available as a fallback (`smoothing_method: "moving_average"`)
  7. Publish `TrackedTarget` with normalized center, normalized bbox height (distance proxy), confidence, track_id, `target_visible`, and `stamp`. `target_visible` stays false during occlusion prediction; the controller's 10 Hz watchdog remains the safety authority
  8. When `publish_annotated_image` is true, publish an overlay frame on `/perception/annotated` with bbox, track ID, confidence, FPS, inference time, and smoothing method
- **Outputs**: `/target/pose` (visual_teleop_msgs/TrackedTarget), `/perception/annotated` (sensor_msgs/Image)

### controller_node
- **Input**: `/target/pose` (visual_teleop_msgs/TrackedTarget)
- **Processing**:
  1. Extract target position (x, y normalized, confidence, target_visible, track_id, stamp)
  2. Compute angular error from image center (x - 0.5)
  3. Apply proportional control for angular velocity
  4. Hold `target_distance` using normalized bbox height as a pinhole distance proxy (`d ~= 0.5 * target_distance / h`); allow forward and backward motion with `deadband` in meters
  5. Clamp to `max_linear_speed` / `max_angular_speed`
  6. Apply dead zone (`dead_zone_px`)
  7. Safety (subscription callback + watchdog): publish zero velocity if `target_visible=false` for > `target_lost_timeout_sec` (default 1.0s). The watchdog timer (10 Hz) ensures zero Twist is published even if no new messages arrive.
  8. **Latency logging**: computes and logs time from frame capture (`msg.stamp`) to Twist publish (~every 2s)
- **Output**: `/cmd_vel` (geometry_msgs/TwistStamped)

## Data Flow Timing
Target latency budget: **<150ms camera-to-cmd_vel** (measured via `msg.stamp` → controller publish)

| Stage | Target Latency |
|-------|----------------|
| Camera capture | 0-33ms (latest-frame thread at 30 FPS) |
| YOLO inference | ~15-40ms (CUDA, yolov8n, imgsz=320) |
| ByteTrack update | ~5ms |
| TrackedTarget publish | ~1ms |
| Controller compute | <1ms |
| ROS 2 publish | ~5ms |
| **Total** | **~50-100ms typical** |

Latency is logged by `controller_node` from frame capture timestamp to
`/cmd_vel` publish. GPU/CPU contention (for example, with the Gazebo GUI)
will increase inference time and therefore end-to-end latency.

## Parameters
All parameters in `config/params.yaml`, loaded via `declare_parameter`.

### perception_node
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| camera_index | int | 0 | Video device index (/dev/videoX) |
| camera_source | string | "" | Optional video file path; overrides camera_index |
| camera_width | int | 640 | Capture width |
| camera_height | int | 480 | Capture height |
| camera_fps | int | 30 | Capture FPS |
| camera_backend | string | "v4l2" | Backend: v4l2, any, gstreamer, ffmpeg |
| camera_fourcc | string | "MJPG" | FourCC format: MJPG (YUYV produces corrupted frames in this env) |
| camera_warmup_frames | int | 10 | Frames to discard for auto-exposure/white-balance settle |
| camera_capture_threaded | bool | true | Latest-frame daemon grabber (skip ticks without a new frame) |
| yolo_model | string | "yolov8n.pt" | YOLO model file |
| target_class | string | "person" | COCO class to track |
| confidence_threshold | double | 0.5 | Detection threshold |
| iou_threshold | double | 0.45 | NMS IoU threshold |
| device | string | "auto" | auto → cuda:0 if available, else cpu |
| yolo_imgsz | int | 320 | Inference resolution |
| yolo_half | bool | true | FP16 on CUDA only (`quantize=16`) |
| yolo_max_det | int | 20 | Cap detections before NMS |
| track_activation_threshold | double | 0.25 | Confidence threshold for activating new tracks |
| minimum_matching_threshold | double | 0.8 | IoU threshold for matching detections to tracks |
| max_time_lost | int | 30 | Max frames to keep lost track alive |
| minimum_consecutive_frames | int | 1 | Min consecutive frames to confirm a track |
| smoothing_method | string | "kalman" | kalman, moving_average, or none |
| smoothing_window_size | int | 5 | Moving-average fallback window (0=disabled) |
| kalman_process_noise | double | 0.1 | White-acceleration q (higher = more responsive) |
| kalman_measurement_noise | double | 0.002 | Measurement variance r (higher = smoother) |
| kalman_initial_covariance | double | 1.0 | Initial state uncertainty |
| kalman_coast_frames | int | 15 | Max predict-only frames during occlusion |
| publish_annotated_image | bool | true | Publish /perception/annotated image |
| inference_poll_rate_hz | double | 60.0 | Latest-frame polling rate |
| publish_rate_hz | double | 30.0 | Minimum interval for Kalman dt |

### controller_node
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| linear_gain | double | 0.5 | P gain for linear velocity |
| angular_gain | double | 1.0 | P gain for angular velocity |
| max_linear_speed | double | 0.22 | Max linear speed (m/s) |
| max_angular_speed | double | 1.82 | Max angular speed (rad/s) |
| target_distance | double | 1.0 | Desired follow distance (meters) |
| deadband | double | 0.1 | Distance deadband (meters) |
| dead_zone_px | double | 0.05 | Dead zone in normalized x (0-1) for angular control |
| target_lost_timeout_sec | double | 1.0 | Seconds before stopping if target lost (used by both callback check and watchdog timer) |
| enable_safety_stop | bool | true | Stop robot if no target detected |
| publish_rate_hz | double | 60.0 | cmd_vel Hz |

## Dependencies
- **Core**: rclpy, std_msgs, sensor_msgs, geometry_msgs, cv_bridge
- **Perception**: ultralytics (YOLO), opencv-python, supervision (ByteTrack)
- **Messages**: visual_teleop_msgs (TrackedTarget with stamp field)
- **Simulation**: turtlebot3_gazebo, ros_gz_bridge, ros_gz_sim
- **Launch**: launch, launch_ros