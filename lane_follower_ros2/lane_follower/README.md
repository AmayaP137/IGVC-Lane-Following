# lane_follower — ROS 2 Jazzy

CNN-based lane detection and following on Raspberry Pi 5.

## Architecture

```
/dev/video0
     │
     ▼
[camera_node]  ──────────►  /camera/image_raw  (sensor_msgs/Image)
                                     │
                                     ▼
                           [detector_node]  (ONNX inference)
                           │              │
                           ▼              ▼
                      /lane/mask     /lane/centroid
                    (mono8 mask)   (PointStamped x,y)
                                        │
                                        ▼
                              [controller_node]  (PD control)
                                        │
                                        ▼
                                   /cmd_vel  (Twist)
```

## Quick Start

### 1. Install dependencies
```bash
sudo apt install ros-jazzy-cv-bridge ros-jazzy-image-transport
pip3 install onnxruntime opencv-python numpy
```

### 2. Copy your ONNX model
```bash
mkdir -p ~/ros2_ws/src/lane_follower/models
cp /path/to/lane_detector.onnx ~/ros2_ws/src/lane_follower/models/
```
Update `model_path` in `config/params.yaml` if needed.

### 3. Build & source
```bash
cd ~/ros2_ws
colcon build --packages-select lane_follower
source install/setup.bash
```

### 4. Launch
```bash
ros2 launch lane_follower lane_follower.launch.py
```

## Tuning the PD Controller

| Parameter | Effect |
|---|---|
| `kp` ↑ | Faster correction, but may oscillate |
| `kd` ↑ | Dampens oscillation |
| `base_speed` ↑ | Faster forward — reduce for Pi 5 safety |
| `roi_top_frac` ↑ | Ignore more of the top of frame (remove horizon) |

**Start with `base_speed: 0.10` and `kp: 0.5` until behaviour is stable.**

## Topics

| Topic | Type | Description |
|---|---|---|
| `/camera/image_raw` | sensor_msgs/Image | Raw BGR frames |
| `/lane/mask` | sensor_msgs/Image | 3-class seg mask (0=bg 1=white 2=yellow) |
| `/lane/centroid` | geometry_msgs/PointStamped | Lane centre x,y in image pixels |
| `/cmd_vel` | geometry_msgs/Twist | Drive commands |

## Visualise mask in RViz2
Add an Image display, set topic to `/lane/mask`.

## CSI Camera (Pi Camera Module)
Change `device_index` to the correct V4L2 index, or replace `cv2.VideoCapture`
in `camera_node.py` with a `picamera2` source.
