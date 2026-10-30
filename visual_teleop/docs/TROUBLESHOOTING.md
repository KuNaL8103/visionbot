# Visual Teleoperation Troubleshooting Guide

## Webcam Issues

### Solid green/corrupted frames in both cv2.imshow AND cv2.imwrite

**Symptom**: Camera opens successfully (`cap.isOpened() == True`, `ret == True`, correct frame shape), but:
- `cv2.imshow` shows solid green window (or solid color with thin correct strip at top)
- `cv2.imwrite` saves PNG files that are ALSO solid green when opened in image viewer
- Frame statistics show low unique color count, most pixels identical

**False lead (ruled out)**: Assumed it was a Qt/GTK/WSLg display bug with `cv2.imshow`. **Ruled out** because `cv2.imwrite` bypasses all GUI code entirely (no Qt, no imshow, no window system) and produces identical corruption. If only `imshow` were broken, `imwrite` would save correct frames.

**Root cause**: OpenCV was forcing `CAP_PROP_FOURCC` to YUYV (the camera's default raw format), but this specific webcam (ACER HD User Facing via usbipd-win/WSL2) streams **MJPG-compressed frames by default**. When OpenCV requests YUYV but the camera delivers MJPG bytes, OpenCV misinterprets the JPEG bitstream as raw YUYV pixels, producing the solid-green corruption pattern.

**Evidence**:
- `v4l2-ctl --list-formats-ext` showed MJPG listed as format [0] (first/highest priority) at 640x480@30fps
- Forcing `camera_fourcc: "MJPG"` in params.yaml fixed it immediately
- With MJPG: unique colors jumped from ~1,700 (corrupted) to ~11,000+ (real image), frame mean ~140 vs ~52, std ~53 vs ~75

**Fix**: In `config/params.yaml`, set:
```yaml
camera_fourcc: "MJPG"   # Force MJPG decoding instead of default YUYV
camera_backend: "v4l2"  # Explicit V4L2 backend (required for WSL2 webcam access)
camera_warmup_frames: 10  # Discard frames for auto-exposure/white-balance settle
```

**Verification**: After fix, `cv2.imwrite` PNGs show real video content, `cv2.imshow` debug window shows live feed with YOLO bounding boxes, and `target_visible` toggles correctly with person detection confidence scores.

---

### ByteTrack ID resets after full occlusion

**Symptom**: `track_id` in `/target/pose` changes to a new number after the target person fully leaves the frame (no bounding box at all, `target_visible: false`) and then re-enters. During continuous tracking with partial occlusion (person still partially visible, bounding box present but partly obscured), `track_id` stays the same.

**Cause**: The `supervision.ByteTrack` implementation used here performs **motion-prediction + IoU matching only** — it has **no visual appearance-based re-identification**. When the target fully leaves the frame:
1. Track enters "lost" state
2. After `max_time_lost` frames (default 30), track is purged
3. When person re-enters, it's treated as a new detection → new `track_id`

Even before `max_time_lost`, if the predicted position drifts past the `minimum_matching_threshold` (IoU 0.8), the new detection won't match the old track.

**Accepted limitation**: This project's controller only needs `x`, `y`, and `target_visible` to function — it does **not** depend on persistent identity across full occlusions. Adding appearance-based ReID (e.g., OSNet, DeepSORT) would add significant complexity and latency for no controller benefit.

**Workaround (if ID persistence is needed later)**: Increase `max_time_lost` in params.yaml, or integrate a ReID model (supervision supports `ByteTrack + ReID` via custom extension).

---

### Camera not found /dev/video0
```bash
# Check if camera exists
ls -la /dev/video*
v4l2-ctl --list-devices

# Check kernel module
lsmod | grep uvcvideo
sudo modprobe uvcvideo

# Check dmesg for USB errors
dmesg -T | grep -i usb
```

### OpenCV can't open camera
```bash
# Test with Python
python3 -c "
import cv2
cap = cv2.VideoCapture(0)
print('Opened:', cap.isOpened())
ret, frame = cap.read()
print('Frame:', ret, frame.shape if ret else 'None')
cap.release()
"

# Common fixes:
# 1. Check user in video group
groups $USER
sudo usermod -aG video $USER  # Then logout/login

# 2. Check camera in use by another process
lsof /dev/video0
sudo fuser -k /dev/video0

# 3. Try different backend
python3 -c "
import cv2
for backend in [cv2.CAP_V4L2, cv2.CAP_GSTREAMER, cv2.CAP_FFMPEG]:
    cap = cv2.VideoCapture(0, backend)
    print(f'Backend {backend}: {cap.isOpened()}')
    cap.release()
"
```

### GStreamer warnings (normal)
```
[ WARN:0] global ... cap_gstreamer.cpp (862) isPipelinePlaying OpenCV | GStreamer warning: GStreamer: pipeline have not been created
```
**This is normal** - OpenCV falls back to V4L2 backend automatically.

---

## ROS 2 Issues

### Package not found after build
```bash
# Rebuild with symlink install
colcon build --symlink-install
source install/setup.bash

# Check package registered
ros2 pkg list | grep visual_teleop
```

### Node executable not found
```bash
# Check entry points in setup.py
# Verify console_scripts section

# Rebuild
colcon build --packages-select visual_teleop --symlink-install
```

### Parameter file not loading
```bash
# Verify params.yaml installed
ls install/visual_teleop/share/visual_teleop/config/params.yaml

# Check parameter loading in node
ros2 param list /perception_node
ros2 param get /perception_node camera_index
```

### `ros2 run` does NOT load config/params.yaml automatically

**Issue**: Running `ros2 run visual_teleop perception_node` starts the node but **does not load** `config/params.yaml`. Parameters fall back to `declare_parameter` defaults in the code, not the YAML values. This caused significant confusion during debugging when camera settings appeared ignored.

**Why**: `ros2 run` only loads parameters from the command line (`-p key:=value`). The YAML file is loaded by `ros2 launch` via the `parameters:` field in the launch file, or explicitly with `--params-file`.

**Correct ways to run with params.yaml**:

```bash
# Option 1: Use ros2 launch (loads params.yaml via launch file)
ros2 launch visual_teleop perception.launch.py

# Option 2: Pass params file explicitly to ros2 run
ros2 run visual_teleop perception_node --ros-args --params-file install/visual_teleop/share/visual_teleop/config/params.yaml

# Option 3: Override individual params on command line
ros2 run visual_teleop perception_node --ros-args -p camera_fourcc:=MJPG -p camera_backend:=v4l2
```

**During development**: Always use `ros2 launch` or `--params-file` when testing parameter changes. The launch file `perception.launch.py` already includes the params file.

### Topic not publishing
```bash
# Check topic exists
ros2 topic list | grep target

# Check publisher
ros2 topic info /target/pose

# Echo topic
ros2 topic echo /target/pose --once
```

---

## Perception Issues

### YOLO model download fails
```bash
# Manual download
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O ~/.local/share/ultralytics/yolov8n.pt

# Or let ultralytics auto-download (needs internet first run)
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### Low FPS / High latency
```bash
# Check CPU usage
htop

# Reduce resolution in params.yaml
camera_width: 320
camera_height: 240

# Use smaller model
yolo_model: "yolov8n.pt"  # nano is fastest

# Enable CUDA if available
device: "cuda"
```

### No detections / Wrong class
```bash
# List COCO classes
python3 -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print(model.names)"

# Common classes: person(0), cell phone(67), cup(41), bottle(39), hand(not in COCO)
# For hand tracking: use custom model or MediaPipe Hands
```

### ByteTrack not tracking
```bash
# Check tracker parameters in params.yaml
tracking_max_age: 30      # Increase if target moves fast
tracking_min_hits: 3      # Decrease for faster confirmation
tracking_iou_threshold: 0.3  # Lower for more permissive matching
```

---

## Controller Issues

### Robot not moving
```bash
# Check cmd_vel published
ros2 topic echo /cmd_vel

# Check target pose received
ros2 topic echo /target/pose

# Verify controller node running
ros2 node list
ros2 node info /controller_node
```

### Robot oscillates / Unstable
```bash
# Tune gains in params.yaml
linear_gain: 0.3      # Reduce if overshooting
angular_gain: 0.5     # Reduce if wobbling

# Increase deadband
deadband: 0.15

# Check max speeds not too high
max_linear_speed: 0.15
```

### Robot doesn't stop when target lost
```bash
# Verify safety stop enabled
enable_safety_stop: true

# Check timeout
lost_target_timeout: 1.0  # Seconds

# Monitor target pose frequency
ros2 topic hz /target/pose
```

---

## Gazebo / TurtleBot3 Issues

### TurtleBot3 packages not installed
```bash
# Install from apt
sudo apt update
sudo apt install ros-jazzy-turtlebot3* ros-jazzy-gazebo-ros-pkgs

# Or build from source
cd ~/ros2_ws/src
git clone -b jazzy-devel https://github.com/ROBOTIS-GIT/turtlebot3.git
git clone -b jazzy-devel https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git
colcon build
```

### Gazebo fails to start (WSL2)
```bash
# Need X11 forwarding
# Option 1: VcXsrv on Windows + export DISPLAY
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0

# Option 2: Use WSLg (Windows 11 built-in)
# Should work automatically

# Option 3: Headless Gazebo
export GAZEBO_HEADLESS_RENDERING=1
export GZ_SIM_HEADLESS=1
```

### Robot spawns but falls through ground
```bash
# Check world file has ground plane
# Verify collision geometry in URDF
```

---

## Build Issues

### colcon build fails
```bash
# Clean build
rm -rf build install log
colcon build --symlink-install

# Verbose output
colcon build --event-handlers console_direct+

# Build single package
colcon build --packages-select visual_teleop --symlink-install
```

### Python import errors
```bash
# Ensure dependencies installed
pip install ultralytics supervision opencv-python

# Or use rosdep
rosdep install --from-paths src --ignore-src -r -y
```

---

## Performance Tips

1. **Resize input**: 320x240 instead of 640x480 for faster inference
2. **Use yolov8n.pt**: Smallest/fastest YOLOv8 model
3. **Limit publish rate**: Match camera FPS (30 Hz max)
4. **Profile with**: `ros2 topic hz /target/pose` and `ros2 topic hz /cmd_vel`
5. **Monitor latency**: Add timestamp comparison in controller_node

---

## Getting Help

1. Check `ros2 wtf` for system issues
2. Enable debug logging: `ros2 run visual_teleop perception_node --ros-args --log-level DEBUG`
3. Check [ROS 2 Jazzy docs](https://docs.ros.org/en/jazzy/)
4. Check [ultralytics YOLO docs](https://docs.ultralytics.com/)
5. Check [supervision ByteTrack docs](https://supervision.roboflow.com/trackers/byte_tracker/)