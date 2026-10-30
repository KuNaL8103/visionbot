#!/usr/bin/env python3
"""
Perception Node - YOLO Detection + ByteTrack

Opens the webcam with a low-latency latest-frame capture thread, runs YOLOv8
detection on the newest frame, uses ByteTrack to maintain stable track IDs
across frames, and publishes TrackedTarget with the normalized bbox center of
the highest-confidence detection for the configured target_class.

A constant-velocity Kalman filter smooths x/y and predicts through brief
occlusions. The 5-frame moving average remains available as a fallback.
"""

import os
import threading
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Image
from ultralytics import YOLO

from visual_teleop.utils.kalman_filter import KalmanFilter2D
from visual_teleop.utils.tracker_wrapper import TrackerWrapper
from visual_teleop_msgs.msg import TrackedTarget
from collections import deque


# Low-latency target stream: keep only the newest sample and never queue old
# bounding boxes behind a slow controller/inference cycle.
TARGET_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
)


class PerceptionNode(Node):
    """Node for publishing YOLO-detected target position."""

    @staticmethod
    def _resolve_device(requested_device: str):
        """Resolve 'auto' (or an unavailable CUDA device) to a usable device."""
        requested = (requested_device or 'auto').lower()
        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except Exception:
            cuda_available = False

        if requested == 'auto':
            return ('cuda:0' if cuda_available else 'cpu'), cuda_available
        if requested.startswith('cuda') and not cuda_available:
            return 'cpu', False
        return requested, cuda_available

    def _resolve_capture_source(self):
        """Return (OpenCV source, backend) for a webcam index or video file."""
        backend_map = {
            'v4l2': cv2.CAP_V4L2,
            'any': cv2.CAP_ANY,
            'gstreamer': cv2.CAP_GSTREAMER,
            'ffmpeg': cv2.CAP_FFMPEG,
        }
        backend = backend_map.get(self.camera_backend.lower(), cv2.CAP_V4L2)
        source = (self.camera_source or '').strip()
        if not source:
            return self.camera_index, backend
        if source.lstrip('+-').isdigit():
            return int(source), backend
        return os.path.expanduser(source), cv2.CAP_FFMPEG

    def _inference_kwargs(self):
        """Build YOLO kwargs once per call, including the correct FP16 flag."""
        kwargs = {
            'verbose': False,
            'conf': self.confidence_threshold,
            'iou': self.iou_threshold,
            'device': self.device,
            'imgsz': self.yolo_imgsz,
            'max_det': self.yolo_max_det,
        }
        if self.target_class_idx is not None:
            kwargs['classes'] = [self.target_class_idx]
        if self.use_half:
            # Ultralytics >= 8.4 uses quantize=16; older versions use half=True.
            kwargs[self._fp16_arg] = 16 if self._fp16_arg == 'quantize' else True
        return kwargs

    def _warmup_inference(self):
        """Run one dummy inference so CUDA/JIT/NMS setup cost is paid at startup."""
        if self.model is None:
            return
        dummy = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
        started = time.perf_counter()
        try:
            self.model(dummy, **self._inference_kwargs())
            warmup_ms = (time.perf_counter() - started) * 1000.0
            self.get_logger().info(f'YOLO warmup complete: {warmup_ms:.1f}ms')
        except Exception as exc:
            self.get_logger().warn(f'YOLO warmup skipped: {exc}')

    def __init__(self, parameter_overrides=None):
        super().__init__('perception_node', parameter_overrides=parameter_overrides)

        # Declare parameters (loaded from config/params.yaml)
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('camera_source', '')  # Optional video file path
        self.declare_parameter('camera_width', 640)
        self.declare_parameter('camera_height', 480)
        self.declare_parameter('camera_fps', 30)
        self.declare_parameter('camera_backend', 'v4l2')  # v4l2, any, gstreamer, ffmpeg
        # MJPG (default — YUYV causes corrupted frames on this hardware, see TROUBLESHOOTING.md)
        self.declare_parameter('camera_fourcc', 'MJPG')
        # Frames to discard for auto-exposure settle
        self.declare_parameter('camera_warmup_frames', 10)
        self.declare_parameter('camera_capture_threaded', True)  # Latest-frame grabber
        self.declare_parameter('publish_rate_hz', 30.0)
        self.declare_parameter('yolo_model', 'yolov8n.pt')
        self.declare_parameter('target_class', 'person')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('device', 'auto')  # auto, cpu, cuda, cuda:0, ...
        # Inference resolution (320 is ~3x faster than 640)
        self.declare_parameter('yolo_imgsz', 320)
        self.declare_parameter('yolo_half', True)  # FP16 on CUDA only
        self.declare_parameter('yolo_max_det', 20)  # Cap boxes before NMS
        # Poll latest frame faster than the camera
        self.declare_parameter('inference_poll_rate_hz', 60.0)
        self.declare_parameter('publish_annotated_image', True)
        self.declare_parameter('show_debug_window', False)
        # ByteTrack parameters
        self.declare_parameter('track_activation_threshold', 0.25)
        self.declare_parameter('minimum_matching_threshold', 0.8)
        self.declare_parameter('max_time_lost', 30)
        self.declare_parameter('minimum_consecutive_frames', 1)

        # Smoothing filter parameters
        self.declare_parameter('smoothing_method', 'kalman')  # kalman, moving_average, none
        self.declare_parameter('smoothing_window_size', 5)  # Moving-average window, 0=disabled
        # Constant-velocity Kalman filter (pure NumPy)
        self.declare_parameter('kalman_process_noise', 0.1)        # White-acceleration q
        self.declare_parameter('kalman_measurement_noise', 0.002)  # Measurement variance r
        self.declare_parameter('kalman_initial_covariance', 1.0)   # Initial state uncertainty
        # Max predict-only frames during occlusion
        self.declare_parameter('kalman_coast_frames', 15)

        # Get parameter values
        self.camera_index = self.get_parameter('camera_index').get_parameter_value().integer_value
        self.camera_source = self.get_parameter('camera_source').get_parameter_value().string_value
        self.camera_width = self.get_parameter('camera_width').get_parameter_value().integer_value
        self.camera_height = self.get_parameter(
            'camera_height').get_parameter_value().integer_value
        self.camera_fps = self.get_parameter('camera_fps').get_parameter_value().integer_value
        self.camera_backend = self.get_parameter(
            'camera_backend').get_parameter_value().string_value
        self.camera_fourcc = self.get_parameter('camera_fourcc').get_parameter_value().string_value
        self.camera_warmup_frames = self.get_parameter(
            'camera_warmup_frames').get_parameter_value().integer_value
        self.camera_capture_threaded = self.get_parameter(
            'camera_capture_threaded').get_parameter_value().bool_value
        self.publish_rate_hz = self.get_parameter(
            'publish_rate_hz').get_parameter_value().double_value
        self.yolo_model_path = self.get_parameter('yolo_model').get_parameter_value().string_value
        self.target_class = self.get_parameter('target_class').get_parameter_value().string_value
        self.confidence_threshold = self.get_parameter(
            'confidence_threshold').get_parameter_value().double_value
        self.iou_threshold = self.get_parameter('iou_threshold').get_parameter_value().double_value
        requested_device = self.get_parameter('device').get_parameter_value().string_value
        self.yolo_imgsz = self.get_parameter('yolo_imgsz').get_parameter_value().integer_value
        self.yolo_half_requested = self.get_parameter('yolo_half').get_parameter_value().bool_value
        self.yolo_max_det = self.get_parameter('yolo_max_det').get_parameter_value().integer_value
        self.inference_poll_rate_hz = self.get_parameter(
            'inference_poll_rate_hz').get_parameter_value().double_value
        self.publish_annotated_image = (
            self.get_parameter('publish_annotated_image').get_parameter_value().bool_value
        )
        self.device, cuda_available = self._resolve_device(requested_device)
        self.use_half = bool(self.yolo_half_requested and cuda_available)
        self.last_inference_ms = None
        self._fps = 0.0
        self._last_frame_time = None
        self.show_debug_window = self.get_parameter(
            'show_debug_window').get_parameter_value().bool_value
        # ByteTrack parameters
        self.track_activation_threshold = self.get_parameter(
            'track_activation_threshold').get_parameter_value().double_value
        self.minimum_matching_threshold = self.get_parameter(
            'minimum_matching_threshold').get_parameter_value().double_value
        self.max_time_lost = self.get_parameter(
            'max_time_lost').get_parameter_value().integer_value
        self.minimum_consecutive_frames = self.get_parameter(
            'minimum_consecutive_frames').get_parameter_value().integer_value

        # Smoothing filter parameters
        self.smoothing_method = self.get_parameter(
            'smoothing_method').get_parameter_value().string_value.lower()
        self.smoothing_window_size = self.get_parameter(
            'smoothing_window_size').get_parameter_value().integer_value
        self.kalman_process_noise = self.get_parameter(
            'kalman_process_noise').get_parameter_value().double_value
        self.kalman_measurement_noise = self.get_parameter(
            'kalman_measurement_noise').get_parameter_value().double_value
        self.kalman_initial_covariance = self.get_parameter(
            'kalman_initial_covariance').get_parameter_value().double_value
        self.kalman_coast_frames = self.get_parameter(
            'kalman_coast_frames').get_parameter_value().integer_value

        if self.smoothing_method not in ('kalman', 'moving_average', 'none'):
            self.get_logger().warn(
                f'Unknown smoothing_method "{self.smoothing_method}", falling back to "none"')
            self.smoothing_method = 'none'

        # Initialize smoothing buffers (moving-average fallback / comparison mode).
        self.smoothing_buffer_x = None
        self.smoothing_buffer_y = None
        if self.smoothing_method == 'moving_average' and self.smoothing_window_size > 0:
            self.smoothing_buffer_x = deque(maxlen=self.smoothing_window_size)
            self.smoothing_buffer_y = deque(maxlen=self.smoothing_window_size)

        # Constant-velocity Kalman filter (pure NumPy). A single filter is used
        # because the perception node publishes only the best (highest-confidence)
        # target at a time; ByteTrack handles identity across detections.
        self.kalman_filter = None
        if self.smoothing_method == 'kalman':
            filter_dt = 1.0 / self.publish_rate_hz if self.publish_rate_hz > 0.0 else 1.0 / 30.0
            self.kalman_filter = KalmanFilter2D(
                process_noise=self.kalman_process_noise,
                measurement_noise=self.kalman_measurement_noise,
                dt=filter_dt,
                initial_covariance=self.kalman_initial_covariance,
                max_coast_frames=self.kalman_coast_frames,
            )
        self.last_track_id = None

        self.get_logger().info('Perception node initialized')
        self.get_logger().info(f'  camera_index: {self.camera_index}')
        self.get_logger().info(f'  camera_source: {self.camera_source or "<webcam index>"}')
        self.get_logger().info(f'  camera_width: {self.camera_width}')
        self.get_logger().info(f'  camera_height: {self.camera_height}')
        self.get_logger().info(f'  camera_fps: {self.camera_fps}')
        self.get_logger().info(f'  camera_backend: {self.camera_backend}')
        self.get_logger().info(f'  camera_fourcc: {self.camera_fourcc}')
        self.get_logger().info(f'  camera_warmup_frames: {self.camera_warmup_frames}')
        self.get_logger().info(f'  camera_capture_threaded: {self.camera_capture_threaded}')
        self.get_logger().info(f'  publish_rate_hz: {self.publish_rate_hz}')
        self.get_logger().info(f'  yolo_model: {self.yolo_model_path}')
        self.get_logger().info(f'  target_class: {self.target_class}')
        self.get_logger().info(f'  confidence_threshold: {self.confidence_threshold}')
        self.get_logger().info(f'  iou_threshold: {self.iou_threshold}')
        self.get_logger().info(f'  device: {self.device}')
        self.get_logger().info(f'  yolo_imgsz: {self.yolo_imgsz}')
        self.get_logger().info(f'  yolo_half: {self.use_half}')
        self.get_logger().info(f'  yolo_max_det: {self.yolo_max_det}')
        self.get_logger().info(f'  inference_poll_rate_hz: {self.inference_poll_rate_hz}')
        self.get_logger().info(f'  publish_annotated_image: {self.publish_annotated_image}')
        self.get_logger().info(f'  show_debug_window: {self.show_debug_window}')
        self.get_logger().info(f'  track_activation_threshold: {self.track_activation_threshold}')
        self.get_logger().info(f'  minimum_matching_threshold: {self.minimum_matching_threshold}')
        self.get_logger().info(f'  max_time_lost: {self.max_time_lost}')
        self.get_logger().info(f'  minimum_consecutive_frames: {self.minimum_consecutive_frames}')
        self.get_logger().info(f'  smoothing_method: {self.smoothing_method}')
        self.get_logger().info(f'  smoothing_window_size: {self.smoothing_window_size}')
        if self.kalman_filter is not None:
            self.get_logger().info(f'  kalman_process_noise: {self.kalman_process_noise}')
            self.get_logger().info(f'  kalman_measurement_noise: {self.kalman_measurement_noise}')
            self.get_logger().info(
                f'  kalman_initial_covariance: {self.kalman_initial_covariance}')
            self.get_logger().info(f'  kalman_coast_frames: {self.kalman_coast_frames}')

        # Load YOLO model
        self.get_logger().info(f'Loading YOLO model: {self.yolo_model_path}')
        self.model = YOLO(self.yolo_model_path)
        self.model.to(self.device)
        # Fuse Conv+BN layers when supported (typically a small speedup).
        try:
            self.model.fuse()
        except Exception:
            pass
        # Pick the precision kwarg supported by this ultralytics release.
        try:
            from ultralytics.cfg import DEFAULT_CFG_DICT
            self._fp16_arg = 'quantize' if 'quantize' in DEFAULT_CFG_DICT else 'half'
        except Exception:
            self._fp16_arg = 'half'
        # Get class names from model
        self.class_names = self.model.names
        self.get_logger().info(f'YOLO model loaded. Classes: {self.class_names}')

        # Find target class index
        self.target_class_idx = None
        for idx, name in self.class_names.items():
            if name == self.target_class:
                self.target_class_idx = idx
                break
        if self.target_class_idx is None:
            self.get_logger().warn(
                f'Target class "{self.target_class}" not found in model classes: '
                f'{list(self.class_names.values())}')
        else:
            self.get_logger().info(
                f'Target class "{self.target_class}" has index {self.target_class_idx}')

        # Pay the first-call CUDA/JIT/NMS cost before the control loop starts.
        self._warmup_inference()

        # Initialize ByteTrack tracker
        self.tracker = TrackerWrapper(
            track_activation_threshold=self.track_activation_threshold,
            minimum_matching_threshold=self.minimum_matching_threshold,
            max_time_lost=self.max_time_lost,
            minimum_consecutive_frames=self.minimum_consecutive_frames
        )
        self.get_logger().info('ByteTrack tracker initialized')

        # Latest-frame capture state. Only one frame is retained; old frames
        # are overwritten rather than queued.
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._latest_frame_stamp = None
        self._frame_seq = 0
        self._processed_frame_seq = 0
        self._capture_stop = threading.Event()
        self._capture_thread = None

        # Open a webcam index or a video file (for reproducible demos/tests).
        capture_source, backend = self._resolve_capture_source()
        self._is_video_file = isinstance(capture_source, str)
        self.cap = cv2.VideoCapture(capture_source, backend)
        if not self.cap.isOpened():
            self.get_logger().warn(
                f'Could not open camera source "{capture_source}" '
                f'with backend {self.camera_backend}'
            )
        elif self._is_video_file:
            self.get_logger().info(f'  Video source opened: {capture_source}')
        else:
            # Keep only the newest driver buffer. Without this, V4L2 can hand
            # back frames that were captured hundreds of milliseconds ago.
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Set FourCC format
            fourcc = cv2.VideoWriter_fourcc(*self.camera_fourcc)
            self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.camera_fps)

            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            actual_fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_str = "".join([chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)])
            self.get_logger().info(
                f'  Camera opened: {int(actual_width)}x{int(actual_height)} '
                f'@ {actual_fps:.1f} FPS, FourCC: {fourcc_str}'
            )

            # Warmup: discard frames to let auto-exposure/white-balance settle
            self.get_logger().info(f'  Warming up camera ({self.camera_warmup_frames} frames)...')
            for _ in range(self.camera_warmup_frames):
                self.cap.read()
            self.get_logger().info('  Camera warmup complete')

        # Publisher for target info (depth=1 keeps only the newest target).
        self.target_pub = self.create_publisher(TrackedTarget, '/target/pose', TARGET_QOS)

        self.bridge = CvBridge()
        self.annotated_pub = None
        if self.publish_annotated_image:
            self.annotated_pub = self.create_publisher(
                Image, '/perception/annotated', qos_profile_sensor_data
            )

        # State for holding last known position when target lost
        self.last_x = 0.5
        self.last_y = 0.5

        # The capture thread is started from main() so direct node construction
        # (unit tests, embedding) remains synchronous unless explicitly started.
        poll_rate = (
            self.inference_poll_rate_hz
            if self.inference_poll_rate_hz > 0.0
            else self.publish_rate_hz
        )
        self.timer = self.create_timer(1.0 / poll_rate, self.timer_callback)

    def start_capture_thread(self):
        """Start the daemon frame grabber (idempotent)."""
        if self._capture_thread is not None and self._capture_thread.is_alive():
            return
        if not self.cap.isOpened():
            self.get_logger().warn('Not starting capture thread: camera is not open')
            return

        self._capture_stop.clear()
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name='visionbot_camera_capture',
            daemon=True,
        )
        self._capture_thread.start()
        self.get_logger().info('Latest-frame capture thread started')

    def _capture_loop(self):
        """Continuously overwrite the single latest-frame slot."""
        file_period = 1.0 / max(self.camera_fps, 1)
        while not self._capture_stop.is_set():
            ret, frame = self.cap.read()
            if not ret:
                if self._is_video_file:
                    self.get_logger().info('Video source reached end of file')
                    self._capture_stop.set()
                    break
                # Avoid a tight spin if the camera is temporarily unavailable.
                self._capture_stop.wait(0.01)
                continue

            stamp = self.get_clock().now().to_msg()
            with self._frame_lock:
                self._latest_frame = frame
                self._latest_frame_stamp = stamp
                self._frame_seq += 1

            if self._is_video_file:
                # Real-time playback instead of decoding the whole file at once.
                self._capture_stop.wait(file_period)

    def _take_latest_frame(self):
        """Return the newest unprocessed frame, or skip if none arrived."""
        with self._frame_lock:
            if self._latest_frame is None or self._frame_seq == self._processed_frame_seq:
                return False, None, None
            self._processed_frame_seq = self._frame_seq
            return True, self._latest_frame, self._latest_frame_stamp

    def timer_callback(self):
        """Timer callback: process only the newest frame and publish TrackedTarget."""
        capture_thread_running = (
            self._capture_thread is not None and self._capture_thread.is_alive()
        )

        if capture_thread_running:
            ret, frame, frame_stamp = self._take_latest_frame()
            if not ret:
                # No new frame since the previous tick: never reprocess a stale
                # frame just to fill the timer period.
                self.get_logger().debug(
                    'No new camera frame; skipping inference tick',
                    throttle_duration_sec=2.0,
                )
                return
        else:
            # Synchronous fallback for direct node construction/tests and for
            # platforms where the capture thread could not start.
            frame_stamp = self.get_clock().now().to_msg()
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warn(
                    'Failed to read frame from camera',
                    throttle_duration_sec=5.0,
                )
                self._publish_target_visible_false(stamp=frame_stamp)
                return

        frame_height, frame_width = frame.shape[:2]

        now_perf = time.perf_counter()
        if self._last_frame_time is not None:
            frame_dt = now_perf - self._last_frame_time
            if frame_dt > 0.0:
                instant_fps = 1.0 / frame_dt
                self._fps = (
                    instant_fps
                    if self._fps <= 0.0
                    else 0.9 * self._fps + 0.1 * instant_fps
                )
        self._last_frame_time = now_perf

        # Run YOLO inference. The kwargs already include device/imgsz and the
        # correct FP16 flag for this ultralytics version.
        inference_started = time.perf_counter()
        results = self.model(frame, **self._inference_kwargs())
        self.last_inference_ms = (time.perf_counter() - inference_started) * 1000.0
        self.get_logger().info(
            f'Inference: {self.last_inference_ms:.1f}ms ({self.device}, imgsz={self.yolo_imgsz})',
            throttle_duration_sec=5.0,
        )

        # Collect all target class detections
        boxes_list = []
        confidences_list = []
        class_ids_list = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_idx = int(box.cls.item())
                conf = float(box.conf.item())
                # Filter by target class
                if self.target_class_idx is not None and cls_idx != self.target_class_idx:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes_list.append([x1, y1, x2, y2])
                confidences_list.append(conf)
                class_ids_list.append(cls_idx)

        # Update ByteTrack tracker
        if boxes_list:
            boxes_array = np.array(boxes_list, dtype=np.float32)
            confidences_array = np.array(confidences_list, dtype=np.float32)
            class_ids_array = np.array(class_ids_list, dtype=np.int32)
        else:
            boxes_array = np.empty((0, 4), dtype=np.float32)
            confidences_array = np.empty(0, dtype=np.float32)
            class_ids_array = np.empty(0, dtype=np.int32)

        tracked_detections = self.tracker.update(boxes_array, confidences_array, class_ids_array)

        # Prepare and publish message
        msg = TrackedTarget()
        msg.stamp = frame_stamp  # Timestamp for latency measurement
        if len(tracked_detections) > 0:
            # Select best track (highest confidence)
            best_idx = np.argmax(tracked_detections.confidence)
            x1, y1, x2, y2 = tracked_detections.xyxy[best_idx]
            best_confidence = float(tracked_detections.confidence[best_idx])
            track_id = (
                int(tracked_detections.tracker_id[best_idx])
                if tracked_detections.tracker_id is not None
                else 0
            )

            # Compute normalized center and height (0.0-1.0, origin top-left)
            raw_center_x = (x1 + x2) / 2.0 / frame_width
            raw_center_y = (y1 + y2) / 2.0 / frame_height
            normalized_height = (y2 - y1) / frame_height

            # Apply the configured smoothing filter to x/y.
            # The moving-average branch is kept for comparison/fallback and is
            # selected whenever its buffers exist.
            if self.smoothing_buffer_x is not None and self.smoothing_buffer_y is not None:
                self.smoothing_buffer_x.append(raw_center_x)
                self.smoothing_buffer_y.append(raw_center_y)
                smoothed_x = sum(self.smoothing_buffer_x) / len(self.smoothing_buffer_x)
                smoothed_y = sum(self.smoothing_buffer_y) / len(self.smoothing_buffer_y)
                msg.x = float(smoothed_x)
                msg.y = float(smoothed_y)
            elif self.kalman_filter is not None:
                # Reset the motion model if ByteTrack assigns a new identity,
                # so stale velocity from the previous target is not reused.
                if self.last_track_id is not None and track_id != self.last_track_id:
                    self.kalman_filter.reset()
                estimate = self.kalman_filter.update(raw_center_x, raw_center_y)
                msg.x = float(estimate[0])
                msg.y = float(estimate[1])
            else:
                # No smoothing - use raw values
                msg.x = float(raw_center_x)
                msg.y = float(raw_center_y)

            self.last_track_id = track_id

            msg.bbox_height = float(normalized_height)
            msg.confidence = best_confidence
            msg.target_visible = True
            msg.track_id = track_id

            # Update last known position (use raw for hold, smoothed for output)
            self.last_x = msg.x
            self.last_y = msg.y

            self._publish_annotated_image(
                frame, frame_stamp, x1, y1, x2, y2, best_confidence, track_id
            )
        else:
            self._publish_target_visible_false(frame, frame_stamp)
            return

        self.target_pub.publish(msg)

    def _publish_target_visible_false(self, frame=None, stamp=None):
        """Publish TrackedTarget with target_visible=false during an occlusion.

        If the Kalman filter has a state, the position is advanced with a
        predict-only step so downstream consumers/debug overlays can see where
        the target is expected to be. ``target_visible`` stays false and the
        10 Hz safety watchdog still stops the robot after its timeout.
        """
        if self.kalman_filter is not None:
            predicted = self.kalman_filter.predict()
            if predicted is not None:
                self.last_x = float(predicted[0])
                self.last_y = float(predicted[1])

        msg = TrackedTarget()
        msg.x = self.last_x
        msg.y = self.last_y
        msg.confidence = 0.0
        msg.target_visible = False
        msg.track_id = 0
        if stamp is not None:
            msg.stamp = stamp
        self.target_pub.publish(msg)

        if frame is not None:
            self._publish_annotated_image(frame, stamp, no_detection=True)

    def _annotate_frame(self, frame, x1=0, y1=0, x2=0, y2=0,
                        conf=0.0, track_id=0, no_detection=False):
        """Return a copy of the frame with tracking and performance overlays."""
        annotated = frame.copy()

        if not no_detection and x2 > x1 and y2 > y1:
            cv2.rectangle(
                annotated, (int(x1), int(y1)), (int(x2), int(y2)),
                (0, 255, 0), 2,
            )
            label = f'{self.target_class} ID:{track_id} {conf:.2f}'
            cv2.putText(
                annotated, label, (int(x1), max(int(y1) - 10, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            cv2.circle(annotated, center, 5, (0, 0, 255), -1)
        else:
            cv2.putText(
                annotated, 'NO DETECTION', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
            )
            center = (
                int(self.last_x * frame.shape[1]),
                int(self.last_y * frame.shape[0]),
            )
            cv2.circle(annotated, center, 5, (255, 0, 0), -1)
            cv2.putText(
                annotated, 'LAST KNOWN', (center[0] + 10, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1,
            )

        h, w = annotated.shape[:2]
        cv2.line(annotated, (w // 2 - 20, h // 2), (w // 2 + 20, h // 2),
                 (255, 255, 255), 1)
        cv2.line(annotated, (w // 2, h // 2 - 20), (w // 2, h // 2 + 20),
                 (255, 255, 255), 1)

        status = (
            f'FPS: {self._fps:.1f} | Smoother: {self.smoothing_method.upper()} '
            f'| Inference: {self.last_inference_ms or 0.0:.1f}ms'
        )
        cv2.putText(
            annotated, status, (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
        )
        return annotated

    def _publish_annotated_image(self, frame, stamp, x1=0, y1=0, x2=0, y2=0,
                                 conf=0.0, track_id=0, no_detection=False):
        """Publish the annotated frame and optionally show the debug window."""
        annotated = self._annotate_frame(
            frame, x1, y1, x2, y2, conf, track_id, no_detection
        )

        if self.annotated_pub is not None:
            image_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            image_msg.header.stamp = (
                stamp if stamp is not None else self.get_clock().now().to_msg()
            )
            image_msg.header.frame_id = 'camera'
            self.annotated_pub.publish(image_msg)

        if self.show_debug_window:
            cv2.imshow('YOLO Detection Debug', annotated)
            cv2.waitKey(1)

    def destroy_node(self):
        # Stop and join the grabber before touching the capture object.
        self._capture_stop.set()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=1.0)
            self._capture_thread = None
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        if self.show_debug_window:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    # Start the grabber only for real execution (not for tests embedding the
    # node directly). `ros2 run` also gets it even without params.yaml.
    if node.camera_capture_threaded:
        node.start_capture_thread()
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
