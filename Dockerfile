# VisionBot — ROS 2 Jazzy development / CI image
#
# Build:
#   docker build -t visionbot:jazzy .
#
# Run a shell with the workspace sourced automatically:
#   docker run --rm -it visionbot:jazzy
#
# Run the test suite headless:
#   docker run --rm visionbot:jazzy bash -lc \
#     "source /opt/ros/jazzy/setup.bash && cd /ros2_ws && \
#      colcon test --packages-select visual_teleop --python-testing pytest \
#      --event-handlers console_direct+ --return-code-on-test-failure && \
#      colcon test-result --verbose"

FROM ros:jazzy-ros-base

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=jazzy

# System + ROS dependencies needed to build the custom message package and
# run the Python tests. python3-opencv keeps cv2 available without pip wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-numpy \
        python3-opencv \
        python3-pytest \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        ros-jazzy-rosidl-default-generators \
        ros-jazzy-builtin-interfaces \
        ros-jazzy-std-msgs \
        ros-jazzy-geometry-msgs \
        ros-jazzy-sensor-msgs \
        ros-jazzy-cv-bridge \
        ros-jazzy-launch-testing \
        ros-jazzy-launch-testing-ament-cmake \
        ros-jazzy-ament-copyright \
        ros-jazzy-ament-flake8 \
        ros-jazzy-ament-pep257 \
    && rm -rf /var/lib/apt/lists/*

# Runtime ML dependencies. Pin versions for reproducible CI builds and keep
# NumPy < 2 to match the ROS 2 Jazzy / OpenCV binary stack.
RUN python3 -m pip install --no-cache-dir --break-system-packages \
        "numpy<2" \
        "opencv-python<4.12" \
        "ultralytics==8.4.142" \
        "supervision==0.30.2" \
        "flake8==7.0.0" \
        "pytest-cov==5.0.0"

WORKDIR /ros2_ws

# The repository root *is* the colcon src directory.
COPY . /ros2_ws/src

# Keep build artifacts out of the source tree and make the installed workspace
# immediately usable in interactive shells.
RUN source /opt/ros/jazzy/setup.bash \
    && colcon build --symlink-install \
        --packages-select visual_teleop_msgs visual_teleop \
        --event-handlers console_direct+ \
    && echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc \
    && echo "source /ros2_ws/install/setup.bash" >> /root/.bashrc

ENV QT_QPA_PLATFORM=offscreen
ENV SDL_VIDEODRIVER=dummy

CMD ["bash"]
