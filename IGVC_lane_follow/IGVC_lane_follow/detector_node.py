#!/usr/bin/env python3
"""
detector_node.py
================
Subscribes to /camera/image_raw, runs the ONNX lane-segmentation model,
and publishes:
  /lane/mask      (sensor_msgs/Image, mono8)   – class map: 0=bg 1=white 2=yellow
  /lane/centroid  (geometry_msgs/PointStamped) – lane-centre x,y in image coords

The centroid message drives controller_node.py.

Parameters:
  model_path     (str)   – absolute path to lane_detector.onnx
  input_width    (int,   default 512)
  input_height   (int,   default 256)
  camera_topic   (str,   default "/camera/image_raw")
  mask_topic     (str,   default "/lane/mask")
  centroid_topic (str,   default "/lane/centroid")
  roi_top_frac   (float, default 0.4) – ignore top N% of image (sky)
"""

import numpy as np
import cv2
import onnxruntime as ort

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge


# ImageNet normalisation constants
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(bgr_frame: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize, normalise, and convert to (1, 3, H, W) float32 tensor."""
    rgb  = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    rgb  = cv2.resize(rgb, (w, h)).astype(np.float32) / 255.0
    rgb  = (rgb - _MEAN) / _STD
    return rgb.transpose(2, 0, 1)[np.newaxis]          # (1, 3, H, W)


def extract_centroid(mask: np.ndarray, roi_top: int) -> tuple[float, float] | None:
    """
    Given a 3-class mask (H×W uint8), find the horizontal midpoint between
    the white lane (class 1) and yellow lane (class 2) centroids in the lower
    portion of the image (below roi_top).

    Returns (cx, cy) in pixel coords, or None if lanes not detected.
    """
    roi_mask = mask[roi_top:, :]

    white_pts  = np.column_stack(np.where(roi_mask == 1))
    yellow_pts = np.column_stack(np.where(roi_mask == 2))

    has_white  = len(white_pts)  > 20
    has_yellow = len(yellow_pts) > 20

    if has_white and has_yellow:
        # Both lanes visible — centre between them
        wx = float(np.mean(white_pts[:, 1]))
        yx = float(np.mean(yellow_pts[:, 1]))
        cx = (wx + yx) / 2.0
        cy = float(roi_top + (mask.shape[0] - roi_top) / 2.0)

    elif has_white:
        # Only white (right lane) — target is to its left
        cx = float(np.mean(white_pts[:, 1])) - mask.shape[1] * 0.15
        cy = float(roi_top + (mask.shape[0] - roi_top) / 2.0)

    elif has_yellow:
        # Only yellow (left lane) — target is to its right
        cx = float(np.mean(yellow_pts[:, 1])) + mask.shape[1] * 0.15
        cy = float(roi_top + (mask.shape[0] - roi_top) / 2.0)

    else:
        return None   # No lanes found

    return cx, cy


class DetectorNode(Node):

    def __init__(self):
        super().__init__('detector_node')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('model_path',     '')
        self.declare_parameter('input_width',    512)
        self.declare_parameter('input_height',   256)
        self.declare_parameter('camera_topic',   '/camera/image_raw')
        self.declare_parameter('mask_topic',     '/lane/mask')
        self.declare_parameter('centroid_topic', '/lane/centroid')
        self.declare_parameter('roi_top_frac',   0.4)

        model_path = self.get_parameter('model_path').value
        self.iw    = self.get_parameter('input_width').value
        self.ih    = self.get_parameter('input_height').value
        cam_topic  = self.get_parameter('camera_topic').value
        mask_topic = self.get_parameter('mask_topic').value
        cent_topic = self.get_parameter('centroid_topic').value
        roi_frac   = self.get_parameter('roi_top_frac').value

        self.roi_top = int(self.ih * roi_frac)

        if not model_path:
            self.get_logger().error(
                'model_path parameter is required. '
                'Set it in config/params.yaml or via --ros-args -p model_path:=/path/to/model.onnx'
            )
            raise ValueError('model_path not set')

        # ── ONNX Runtime ──────────────────────────────────────────────────────
        self.get_logger().info(f'Loading ONNX model from: {model_path}')
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.get_logger().info('Model loaded successfully')

        # ── ROS I/O ───────────────────────────────────────────────────────────
        self.bridge        = CvBridge()
        self.mask_pub      = self.create_publisher(Image,        mask_topic, 10)
        self.centroid_pub  = self.create_publisher(PointStamped, cent_topic, 10)
        self.subscription  = self.create_subscription(
            Image, cam_topic, self.image_callback, 10
        )
        self.get_logger().info(
            f'Detector ready: {cam_topic} → [{mask_topic}, {cent_topic}]'
        )

    def image_callback(self, msg: Image):
        # ── Convert ROS image → numpy BGR ─────────────────────────────────────
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # ── Inference ─────────────────────────────────────────────────────────
        inp    = preprocess(frame, self.ih, self.iw)
        logits = self.session.run(None, {self.input_name: inp})[0]  # (1,3,H,W)
        mask   = logits[0].argmax(axis=0).astype(np.uint8)           # (H,W)

        # ── Publish mask (scaled to 0/127/254 for visualisation) ───────────────
        mask_vis     = mask * 127
        mask_msg     = self.bridge.cv2_to_imgmsg(mask_vis, encoding='mono8')
        mask_msg.header = msg.header
        self.mask_pub.publish(mask_msg)

        # ── Extract & publish lane centroid ───────────────────────────────────
        result = extract_centroid(mask, self.roi_top)
        if result is not None:
            cx, cy = result
            pt                  = PointStamped()
            pt.header           = msg.header
            pt.point.x          = cx
            pt.point.y          = cy
            pt.point.z          = float(self.iw)   # carry image width for normalisation
            self.centroid_pub.publish(pt)
        else:
            self.get_logger().debug('No lanes detected in this frame')


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
