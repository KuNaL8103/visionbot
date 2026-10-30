# VisionBot — Webcam-Driven Visual Teleoperation

A ROS 2 Jazzy project that uses a laptop webcam as the **only sensor** to
track a person and drive a simulated TurtleBot3 robot in Gazebo — no
physical robot hardware required.
 
```
[Webcam] → YOLOv8 detection → ByteTrack tracking → /target/pose
                                                          │
                                                          ▼
                                          Proportional controller
                                                          │
                                                          ▼
                                        /cmd_vel → TurtleBot3 (Gazebo)
```
 

Move in front of the camera, and the simulated robot turns to follow you —
in real time, entirely in simulation.

---

## Table of Contents

- [What This Is](#what-this-is)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Docker & CI](#docker--ci)
- [Running the Full System](#running-the-full-system)
- [Cleaning Up / Restarting](#cleaning-up--restarting)
- [Configuration](#configuration)
- [Message & Topic Reference](#message--topic-reference)
- [Running Tests](#running-tests)
- [Troubleshooting](#troubleshooting)
- [Known Limitations](#known-limitations)
- [Possible Extensions](#possible-extensions)
---

## What This Is

VisionBot is a visual teleoperation pipeline built entirely in software —
your laptop's built-in or USB webcam is the only "sensor" needed. It
demonstrates a full perception-to-control robotics loop:

1. **Perception**: A webcam feed is run through YOLOv8 to detect a person,
   and ByteTrack assigns a stable ID to that detection across frames.
2. **Control**: A proportional controller converts the target's position in
   the camera frame into linear/angular velocity commands.
3. **Simulation**: Those commands drive a simulated TurtleBot3 (Burger)
   robot inside Gazebo, following the ROS 2 standard `/cmd_vel` interface.
This is a portfolio-style project intended to demonstrate real-world ROS 2
skills: custom message design, node architecture, parameter management,
launch file composition, and integrating a computer-vision pipeline with a
robotics control loop and simulator — while working through the genuine
environment quirks of running ROS 2 + Gazebo + a webcam inside WSL2.

## How It Works

### Node Graph
 
```
┌──────────────────┐     /target/pose      ┌──────────────────┐
│  perception_node │ ────────────────────▶ │  controller_node │
│  (OpenCV + YOLO  │  (TrackedTarget)      │  (P Controller)  │
│   + ByteTrack)   │                       └────────┬─────────┘
└──────────────────┘                                 │
                                                      │ /cmd_vel
                                                      │ (TwistStamped)
                                                      ▼
                                             ┌──────────────────┐
                                             │  TurtleBot3      │
                                             │  (Gazebo Sim)    │
                                             └──────────────────┘
```
 

- **`perception_node`** opens the webcam directly via `cv2.VideoCapture`
  (no separate camera driver node). It runs YOLOv8 (`yolov8n.pt`) to detect
  the target class (default: `person`), wraps detections with ByteTrack for
  a stable `track_id` across frames, applies a constant-velocity 2D Kalman
  filter to smooth jitter and predict through brief occlusions, and publishes
  a custom `TrackedTarget` message.
- **`controller_node`** subscribes to that target position, computes a
  horizontal error from image center, and applies proportional control to
  produce linear/angular velocity — clamped to TurtleBot3's real-world
  speed limits. It publishes `geometry_msgs/TwistStamped` (required by this
  project's Gazebo bridge configuration — see
  [Troubleshooting](#troubleshooting)). It also runs a watchdog timer that
  guarantees the robot stops if the target is lost, even if no new
  perception messages arrive.
- **Gazebo** runs the standard TurtleBot3 Burger model in an empty world,
  bridged to ROS 2 topics via `ros_gz_bridge`.
### Control Logic

- **Angular velocity**: `angular_vel = -error_x * angular_gain`, where
  `error_x = target.x - 0.5`. Per ROS convention (REP 103), positive
  `angular.z` is counter-clockwise (turn left). A target on the left side
  of the frame produces positive `angular.z` (robot turns left to face it);
  a target on the right produces negative `angular.z`.
- **Linear velocity**: moves forward at `linear_gain` (clamped to
  `max_linear_speed`) only when the target is visible and roughly centered
  (`|error_x| < 0.3`).
- **Safety stop**: zero velocity is published whenever `target_visible` is
  false, the target hasn't been seen recently, or a 10 Hz watchdog timer
  detects the target has been lost for longer than
  `target_lost_timeout_sec`.

### Latency path

- The camera is opened with `CAP_PROP_BUFFERSIZE=1`; a daemon thread keeps
  only the newest frame, and the inference timer skips if no new frame arrived.
- YOLO runs at `imgsz=320` and FP16 (`half=True`) automatically on CUDA.
- `/target/pose` uses `KEEP_LAST` / `depth=1` QoS, so stale bounding boxes are
  dropped instead of queued; the controller also rejects out-of-order and
  stale frame timestamps.
- `Ctrl+C` is handled cleanly via `ExternalShutdownException` and a guarded
  `rclpy.shutdown()`.

## Project Structure

This repository **is** the colcon workspace's `src/` directory — clone it
directly into `~/ros2_ws/src/` (see [Setup](#setup)). The two ROS 2 packages
sit at the top level of the repo:

```
VisionBot/                                # this repo, clone into ~/ros2_ws/src/
├── .github/workflows/ros2_ci.yml         # Build + headless test CI (Docker)
├── .dockerignore
├── Dockerfile                            # ROS 2 Jazzy dev/CI image
├── RESUME_NOTES.md                       # Kalman/CI interview + resume notes
├── .gitignore
├── README.md                             # this file
├── clean_launch.sh                       # process cleanup helper (see below)
├── visual_teleop/                        # Main ROS 2 package
│   ├── CLAUDE.md                         # Living project-status doc (not tracked)
│   ├── LICENSE                           # Apache 2.0
│   ├── package.xml
│   ├── setup.py / setup.cfg
│   ├── resource/visual_teleop            # ament index marker
│   ├── visual_teleop/                    # Python module
│   │   ├── __init__.py
│   │   ├── perception_node.py            # Webcam → YOLO → ByteTrack → /target/pose
│   │   ├── controller_node.py            # /target/pose → /cmd_vel
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── kalman_filter.py          # Pure-NumPy constant-velocity 2D KF
│   │       └── tracker_wrapper.py        # ByteTrack wrapper (supervision lib)
│   ├── launch/
│   │   ├── perception.launch.py          # Perception node standalone
│   │   ├── sim_turtlebot.launch.py       # TurtleBot3 Gazebo sim standalone
│   │   └── full_system.launch.py         # Everything wired together
│   ├── config/
│   │   └── params.yaml                   # All tunable parameters
│   ├── test/
│   │   ├── test_perception_node.py       # Mocked unit tests (no camera needed)
│   │   ├── test_controller_node.py       # Mocked unit tests (no sim needed)
│   │   ├── test_kalman_filter.py         # Trajectory smoothing / occlusion tests
│   │   └── test_low_latency.py           # Frame-slot / QoS / stale-stamp tests
│   └── docs/
│       ├── ARCHITECTURE.md               # Node graph, data flow, parameters
│       ├── TOPICS.md                     # Message/topic contracts
│       └── TROUBLESHOOTING.md            # Environment issues & fixes
└── visual_teleop_msgs/                   # Custom message package
    ├── CMakeLists.txt
    ├── package.xml
    └── msg/
        └── TrackedTarget.msg             # x, y, confidence, target_visible, track_id, stamp
```

> **Why the packages are at the top level, not under `src/`**: This is the
> standard colcon layout — `colcon build` expects to be run from the
> workspace root (`~/ros2_ws/`) with the packages immediately inside
> `src/`. The repo *is* the `src/` directory.

## Prerequisites

- **OS**: Ubuntu 24.04 (tested on WSL2 under Windows 11, but should work
  the same on native Ubuntu)
- **ROS 2 Jazzy**
- **Python 3.12**
- **A webcam** (built-in laptop cam or USB) — if using WSL2, this needs
  USB passthrough via `usbipd-win`; see [Troubleshooting](#troubleshooting).
- **Gazebo** (Harmonic, ships with ROS 2 Jazzy) + TurtleBot3 packages
### Python Dependencies

```bash
pip install ultralytics opencv-python supervision --break-system-packages
```

- `ultralytics` — YOLOv8 object detection (auto-downloads `yolov8n.pt` on
  first run)
- `opencv-python` — webcam capture and image processing
- `supervision` — ByteTrack multi-object tracking wrapper
### ROS 2 / System Dependencies

```bash
sudo apt update
sudo apt install ros-jazzy-turtlebot3 ros-jazzy-turtlebot3-gazebo
```

## Setup

1. **Clone into a ROS 2 workspace:**
```bash
   mkdir -p ~/ros2_ws/src
   cd ~/ros2_ws/src
   git clone https://github.com/KuNaL8103/VisionBot.git .
```
   The trailing `.` clones the repo *into* the current directory
   (`~/ros2_ws/src/`) instead of creating a `VisionBot/` subfolder. This
   keeps the packages at the top level of `src/` as `colcon` expects.

2. **Install dependencies** (see [Prerequisites](#prerequisites) above).
3. **Build the workspace:**
```bash
   cd ~/ros2_ws
   colcon build --packages-select visual_teleop_msgs visual_teleop
   source install/setup.bash
```

4. **If on WSL2, attach your webcam** (see
   [Troubleshooting](#troubleshooting) for the full usbipd-win flow):
```powershell
   # In Windows PowerShell (Admin)
   usbipd list
   usbipd attach --wsl --busid <BUSID>
```
   Then confirm it's visible inside WSL2:
```bash
   ls -la /dev/video*
```

5. **Set the TurtleBot3 model** (required every session, or add to your
   shell profile):
```bash
   export TURTLEBOT3_MODEL=burger
```

6. **Make the cleanup script executable** (one-time setup — see
   [Cleaning Up / Restarting](#cleaning-up--restarting) for why this
   exists):
```bash
   chmod +x ~/ros2_ws/src/clean_launch.sh
```

## Docker & CI

The repository ships a `Dockerfile` based on `ros:jazzy-ros-base` that
installs the ROS build/test dependencies plus the runtime ML stack
(Ultralytics, Supervision, OpenCV, NumPy). It builds both packages and sources
the workspace automatically:

```bash
docker build -t visionbot:jazzy .
docker run --rm -it visionbot:jazzy
```

Run the full test suite headless inside the container:

```bash
docker run --rm visionbot:jazzy bash -lc \
  "source /opt/ros/jazzy/setup.bash && cd /ros2_ws && \
   colcon test --packages-select visual_teleop --python-testing pytest \
   --event-handlers console_direct+ --return-code-on-test-failure && \
   colcon test-result --verbose"
```

`.github/workflows/ros2_ci.yml` does the same on every push / pull request:
it builds the image with Buildx layer caching, runs `colcon build`, then
executes the **33-test** suite headlessly with `colcon test` and fails the
job on any test failure.

## Running the Full System

The simplest way to run everything — Gazebo sim, perception, and
controller — together:

```bash
export TURTLEBOT3_MODEL=burger
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch visual_teleop full_system.launch.py
```

Wait for Gazebo to fully load (a few seconds), then stand in front of your
webcam. Moving left/right should turn the simulated robot to follow you.

### Running Components Individually (for debugging)

```bash
# Just the Gazebo sim
ros2 launch visual_teleop sim_turtlebot.launch.py

# Just perception, with a debug window showing detections
ros2 run visual_teleop perception_node --ros-args -p show_debug_window:=true

# Just the controller
ros2 run visual_teleop controller_node
```

> **Note**: `ros2 run` does **not** automatically load `config/params.yaml`
> — only `ros2 launch` does. When running a node standalone with `ros2 run`
> for debugging, pass parameters explicitly:
> ```bash
> ros2 run visual_teleop perception_node --ros-args --params-file \
>   install/visual_teleop/share/visual_teleop/config/params.yaml
> ```

### Manually Driving the Robot (sanity check)

```bash
ros2 run turtlebot3_teleop teleop_keyboard
```
(requires `TURTLEBOT3_MODEL=burger` set and the sim already running)

## Cleaning Up / Restarting

Stopping `full_system.launch.py` with **`Ctrl+C` in the terminal that
launched it** is the reliable way to shut everything down cleanly. Closing
the Gazebo window directly, a crash, or killing the terminal itself can
leave orphaned `gz sim`, `perception_node`, or `controller_node` processes
running in the background — which then compete with the next launch for
the webcam, `/cmd_vel`, and `/target/pose`, causing confusing, inconsistent
behavior (e.g. the robot ignoring commands, or a blank/stuck Gazebo window).

As a safety net, `clean_launch.sh` (in the repo root) force-kills any
leftover processes before you launch again:

```bash
#!/bin/bash
echo "Cleaning up any leftover processes..."
pkill -9 -f "gz sim" 2>/dev/null
pkill -9 -f "ros_gz" 2>/dev/null
pkill -9 -f "parameter_bridge" 2>/dev/null
pkill -9 -f "robot_state_publisher" 2>/dev/null
pkill -9 -f "perception_node" 2>/dev/null
pkill -9 -f "controller_node" 2>/dev/null
sleep 2
echo "Done. Process list:"
ps aux | grep -E "gz sim|ros_gz|perception_node|controller_node" | grep -v grep
```

Run it before every launch, especially after a crash or an ungraceful
shutdown:

```bash
~/ros2_ws/src/clean_launch.sh
```

If the process-list output at the end is empty, you're clean. Then launch
normally:

```bash
export TURTLEBOT3_MODEL=burger
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch visual_teleop full_system.launch.py
```

**Optional convenience:** add an alias so you don't need to type the full
path every time:
```bash
echo 'alias clean_launch="~/ros2_ws/src/clean_launch.sh"' >> ~/.bashrc
source ~/.bashrc
# then just run: clean_launch
```

## Configuration

All tunable parameters live in [`config/params.yaml`](visual_teleop/config/params.yaml),
loaded via `declare_parameter` in each node (no hardcoded values in source).
The launch files pick it up automatically.

### Perception node (`perception_node`)

| Parameter | Default | Description |
| --- | --- | --- |
| `camera_index` | `0` | Video device index (`/dev/videoX`) |
| `camera_width` | `640` | Capture width in pixels |
| `camera_height` | `480` | Capture height in pixels |
| `camera_fps` | `30` | Capture FPS requested from the device |
| `camera_backend` | `"v4l2"` | OpenCV backend — `v4l2` required for WSL2 webcam access |
| `camera_fourcc` | `"MJPG"` | Pixel format — **do not change to YUYV**, see Troubleshooting |
| `camera_warmup_frames` | `10` | Frames discarded at startup so auto-exposure/white-balance can settle |
| `camera_capture_threaded` | `true` | Latest-frame daemon grabber; only the newest frame is processed |
| `yolo_model` | `"yolov8n.pt"` | YOLO weights file (auto-downloaded on first run) |
| `target_class` | `"person"` | COCO class to track |
| `confidence_threshold` | `0.5` | Minimum detection confidence to publish |
| `iou_threshold` | `0.45` | NMS IoU threshold |
| `device` | `"auto"` | `auto` → `cuda:0` if available, otherwise `cpu`; or force `cpu` / `cuda:0` |
| `yolo_imgsz` | `320` | Inference resolution; 320 is ~3× faster than 640 |
| `yolo_half` | `true` | FP16 on CUDA (`quantize=16`); automatically disabled on CPU |
| `yolo_max_det` | `20` | Cap detections before NMS |
| `inference_poll_rate_hz` | `60.0` | Poll for the newest frame faster than the 30 FPS camera |
| `track_activation_threshold` | `0.25` | Confidence threshold for activating a new ByteTrack track |
| `minimum_matching_threshold` | `0.8` | IoU threshold for matching detections to existing tracks |
| `max_time_lost` | `30` | Frames a lost track is kept alive before being dropped |
| `minimum_consecutive_frames` | `1` | Consecutive frames required to confirm a track |
| `smoothing_method` | `"kalman"` | `kalman`, `moving_average`, or `none` |
| `smoothing_window_size` | `5` | Moving-average fallback window (0 = disabled) |
| `kalman_process_noise` | `0.1` | White-acceleration `q` — higher is more responsive, less smooth |
| `kalman_measurement_noise` | `0.002` | Measurement variance `r` — higher is smoother, more lag |
| `kalman_initial_covariance` | `1.0` | Initial uncertainty for `x, y, vx, vy` |
| `kalman_coast_frames` | `15` | Predict-only frames during occlusion (~0.5 s at 30 Hz) |
| `publish_annotated_image` | `true` | Publish `/perception/annotated` image |
| `publish_rate_hz` | `30.0` | Publish rate |

> `perception_node` also accepts a boolean `show_debug_window` parameter
> (Python default `false`) that opens an OpenCV window with bounding boxes
> for visual debugging. It is **not** in `config/params.yaml` because it is
> a debug-only switch — pass it on the command line:
> `ros2 run visual_teleop perception_node --ros-args -p show_debug_window:=true`.

### Controller node (`controller_node`)

| Parameter | Default | Description |
| --- | --- | --- |
| `linear_gain` | `0.5` | Proportional gain for linear velocity |
| `angular_gain` | `1.0` | Proportional gain for angular velocity |
| `max_linear_speed` | `0.22` | m/s — TurtleBot3 Burger's real speed limit |
| `max_angular_speed` | `1.82` | rad/s |
| `target_distance` | `1.0` | Desired distance from target (meters) |
| `deadband` | `0.1` | Distance deadband (meters) around `target_distance` |
| `dead_zone_px` | `0.05` | Dead zone in normalized x (0–1) for angular control |
| `target_lost_timeout_sec` | `1.0` | Seconds before stopping if target is lost (used by both the callback and the watchdog timer) |
| `enable_safety_stop` | `true` | Stop the robot if no target is detected |
| `publish_rate_hz` | `60.0` | cmd_vel publish rate (faster than perception updates) |

See [`docs/ARCHITECTURE.md`](visual_teleop/docs/ARCHITECTURE.md) for how the
controller uses these parameters in the control loop.

## Message & Topic Reference

| Topic | Type | Publisher → Subscriber |
| --- | --- | --- |
| `/target/pose` | `visual_teleop_msgs/TrackedTarget` | `perception_node` → `controller_node` |
| `/cmd_vel` | `geometry_msgs/TwistStamped` | `controller_node` → TurtleBot3 (Gazebo) |

**`TrackedTarget.msg`**
```text
float32 x                          # normalized 0.0–1.0, target center, image left→right
float32 y                          # normalized 0.0–1.0, target center, image top→bottom
float32 confidence                 # YOLO detection confidence
bool target_visible                # false when no detection this frame
int32 track_id                     # ByteTrack ID (resets after full occlusion — see docs/TROUBLESHOOTING.md)
builtin_interfaces/Time stamp      # frame capture time, for latency measurement
```
 

Full contract details (QoS, special values, coordinate conventions) are in
`docs/TOPICS.md`.

## Running Tests

All **33** tests are unit tests with mocked hardware — **no camera or Gazebo
required**:

```bash
cd ~/ros2_ws
source install/setup.bash
python3 -m pytest src/visual_teleop/test/ -v
```

Perception tests mock `cv2.VideoCapture`, YOLO, and ByteTrack. Controller
tests use fabricated `TrackedTarget` messages to verify control-loop logic
(sign conventions, clamping, dead zones, watchdog behavior, and latency
computation) in isolation. The Kalman tests verify constant-velocity
convergence, prediction during occlusion, coast limits, reset behavior, and
**30.0% jitter reduction (51.9% after warm-up)** versus the 5-frame moving
average on a seeded synthetic track. Low-latency regression tests cover the
latest-frame slot, auto device resolution, stale frame rejection, and
out-of-order/occluded target handling.

## Troubleshooting

The most valuable lessons from building this project are captured in
`docs/TROUBLESHOOTING.md`. Highlights:

- **Solid green / corrupted webcam frames**: caused by OpenCV defaulting to
  the `YUYV` pixel format while the camera actually streams `MJPG`. Fixed
  by explicitly forcing `camera_fourcc: "MJPG"`. If you see this, don't
  trust `cap.isOpened()`/`ret==True` alone — verify with a saved frame
  (`cv2.imwrite`) that the image is actually real.
- **WSL2 webcam access**: requires `usbipd-win` passthrough from Windows;
  it does not attach automatically.
- **`ros2 run` vs `ros2 launch`**: `ros2 run` does not load
  `config/params.yaml` — use `ros2 launch` or pass `--params-file`
  explicitly.
- **`/cmd_vel` message type**: this project's `ros_gz_bridge` configuration
  only subscribes to `geometry_msgs/TwistStamped`, not plain `Twist` —
  publishing plain `Twist` will silently produce no robot motion.
- **Slow/laggy Gazebo rendering under WSL2**: often caused by falling back
  to CPU software rendering (`llvmpipe`). Check with
  `glxinfo | grep "OpenGL renderer"` — if it shows `llvmpipe`, that's CPU
  rendering. Forcing GPU rendering via Mesa's D3D12 backend helps:
```bash
  export GALLIUM_DRIVER=d3d12
```
  On this project's hardware (RTX 4050 + AMD integrated GPU laptop), this
  alone renders via the **AMD integrated GPU** and is stable. Additionally
  forcing the discrete NVIDIA GPU with
  `MESA_D3D12_DEFAULT_ADAPTER_NAME="NVIDIA GeForce RTX 4050 Laptop GPU"`
  was tested and **reliably crashes Gazebo's GUI** on OpenGL context
  creation (`glx: failed to create drisw screen` → `Aborted (core dumped)`)
  — even though `glxinfo` reports the NVIDIA adapter as working fine. This
  appears specific to Gazebo's Qt/QML rendering path, not a general
  driver problem. **Do not set `MESA_D3D12_DEFAULT_ADAPTER_NAME` to the
  NVIDIA GPU** — leave `GALLIUM_DRIVER=d3d12` set alone (defaults to the
  AMD GPU) for stable, still GPU-accelerated rendering. If your own
  hardware doesn't have this issue, forcing the discrete GPU may work
  fine — test in a throwaway terminal (`export`, not `~/.bashrc`) before
  committing to it.
- **Orphaned processes after a crash or closing the Gazebo window**: see
  [Cleaning Up / Restarting](#cleaning-up--restarting) — use
  `clean_launch.sh` before relaunching.

See `docs/TROUBLESHOOTING.md` for full detail and additional issues
(ByteTrack ID behavior, build errors, etc).

## Known Limitations

- **ByteTrack has no visual re-identification.** `track_id` stays stable
  through brief partial occlusion, but resets to a new ID after the target
  fully leaves the frame and reappears. This doesn't affect the follow
  behavior, since the controller only depends on `x`/`y`/`target_visible`,
  not on ID persistence.
- **Simulation only.** This project has not been tested on physical
  TurtleBot3 hardware; only the Gazebo simulation.
- **Single-target tracking.** If multiple people are in frame, the
  highest-confidence detection is followed; there's no explicit
  multi-person disambiguation logic.
- **TurtleBot3 Burger's real-world top speed is genuinely slow**
  (0.22 m/s) — this is expected, not a performance bug.
## Possible Extensions

- **MoveIt arm variant**: reuse `/target/pose` to drive a robotic arm's
  end-effector instead of a mobile base (`arm_controller_node.py`,
  following a MoveIt demo config).
- **Appearance-based re-identification** to preserve `track_id` across full
  occlusion (e.g., a DeepSORT-style embedding model).
- **Physical hardware deployment** on a real TurtleBot3.
- **Gesture-based control** (e.g., hand gestures instead of person-position
  tracking) using MediaPipe hand landmarks.

*Built as a ROS 2 + computer vision portfolio project — real webcam,
real YOLO detection, real ByteTrack tracking, real proportional control,
real Gazebo simulation.*