#!/usr/bin/env python3
"""
camera_node.py
==============
Wraps a USB / CSI / V4L2 camera (via OpenCV) and publishes
sensor_msgs/Image on /camera/image_raw at a configurable frame rate.

Parameters (set in config/params.yaml or via --ros-args -p):
  device_index  (int,  default 0)   – V4L2 device index (/dev/video0)
  frame_rate    (float, default 30.0)
  image_width   (int,  default 640)
  image_height  (int,  default 480)
  topic         (str,  default "/camera/image_raw")
"""

import cv2
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class CameraNode(Node):

    def __init__(self):
        super().__init__('camera_node')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('device_index', 0)
        self.declare_parameter('frame_rate',   30.0)
        self.declare_parameter('image_width',  640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('topic',        '/camera/image_raw')

        device   = self.get_parameter('device_index').value
        fps      = self.get_parameter('frame_rate').value
        width    = self.get_parameter('image_width').value
        height   = self.get_parameter('image_height').value
        topic    = self.get_parameter('topic').value

        # ── Camera setup ──────────────────────────────────────────────────────
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.get_logger().error(
                f'Cannot open camera at /dev/video{device}. '
                'Check device index and permissions (add user to "video" group).'
            )
            raise RuntimeError('Camera not available')

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS,          fps)

        actual_w   = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h   = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.get_logger().info(
            f'Camera opened: {actual_w}×{actual_h} @ {actual_fps:.1f} fps'
        )

        # ── Publisher & timer ─────────────────────────────────────────────────
        self.bridge     = CvBridge()
        self.publisher  = self.create_publisher(Image, topic, 10)
        self.timer      = self.create_timer(1.0 / fps, self.timer_callback)
        self.frame_id   = 0

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Dropped frame — camera read failed')
            return

        msg               = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp  = self.get_clock().now().to_msg()
        msg.header.frame_id = f'camera_frame_{self.frame_id}'
        self.frame_id    += 1
        self.publisher.publish(msg)

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
