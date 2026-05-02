import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Float32

from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
import sys
import os

from ament_index_python.packages import get_package_share_directory

# --- LINK TO LANENET REPO ---
REPO_PATH = os.path.expanduser('~/ros2_ws/src/lanenet-lane-detection-pytorch')
sys.path.insert(0, REPO_PATH)

from model.lanenet.LaneNet import LaneNet
# from config import global_config


class LaneDetectorNode(Node):

    def __init__(self):
        super().__init__('lane_detector_node')

        # ---------------- PARAMETERS ----------------
        self.declare_parameter('image_topic', '/camera_1/image_raw')
        self.declare_parameter('publish_debug_image', True)

        image_topic = self.get_parameter('image_topic').value
        self.publish_debug = self.get_parameter('publish_debug_image').value

        # ---------------- SETUP ----------------
        self.bridge = CvBridge()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.get_logger().info(f'Running on: {self.device}')

        # ---------------- LOAD MODEL ----------------
        package_share = get_package_share_directory('lanenet_ros')
        weights_path = os.path.join(package_share, 'weights', 'tusimple_lanenet.pth')

       # , cfg=global_config.CFG
       # phase='test'
        self.model = LaneNet()
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # ---------------- ROS INTERFACES ----------------
        self.subscription = self.create_subscription(
            Image, image_topic, self.image_callback, 1)

        # Outputs
        self.mask_pub = self.create_publisher(Image, '/lane/mask', 10)
        self.offset_pub = self.create_publisher(Float32, '/lane/offset', 10)

    # ==========================================================
    # CORE CALLBACK
    # ==========================================================
    def image_callback(self, msg):

    # ---------------- SAFE CONVERSION ----------------
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV bridge failed: {e}")
            return

    # ---------------- PREPROCESS ----------------
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
        cv_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Resize to model input
        resized = cv2.resize(cv_image, (512, 256))

    # Convert to tensor (faster version)
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        tensor = tensor.to(self.device)

    # ---------------- INFERENCE ----------------
        with torch.no_grad():
            output = self.model(tensor)

        binary_seg = output['binary_seg_pred']

    # ---------------- POSTPROCESS ----------------
        binary_seg = binary_seg.squeeze().cpu().numpy()

        mask = (binary_seg > 0.1).astype(np.uint8)

       

    # Morphological cleanup
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # ---------------- FEATURE EXTRACTION ----------------
        lane_offset = self.compute_lane_offset(mask)

    # ---------------- PUBLISH ----------------
        self.offset_pub.publish(Float32(data=float(lane_offset)))

        if self.publish_debug:
            debug_img = (mask * 255).astype(np.uint8)
            out_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding='mono8')
            self.mask_pub.publish(out_msg)

    # ==========================================================
    # LANE GEOMETRY
    # ==========================================================
    def compute_lane_offset(self, mask):
        """
        Compute lateral offset from lane center
        """

        h, w = mask.shape

        # Focus on bottom region (closest to robot)
        roi = mask[int(h * 0.6):, :]

        ys, xs = np.where(roi > 0)

        if len(xs) < 50:
            return 0.0  # no lane detected

        lane_center = np.mean(xs)
        image_center = w / 2

        offset = (image_center - lane_center) / w  # normalize

        return offset


# ==========================================================
# MAIN
# ==========================================================
def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()