#!/usr/bin/env python3

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32MultiArray
from cv_bridge import CvBridge


# ── Steering command labels (for logging only) ─────────────────────────────────
STRAIGHT    = "STRAIGHT"
SOFT_LEFT   = "SOFT LEFT"
HARD_LEFT   = "HARD LEFT"
SOFT_RIGHT  = "SOFT RIGHT"
HARD_RIGHT  = "HARD RIGHT"


class ControllerNode(Node):

    def __init__(self):
        super().__init__('controller_node')

        # misc params
        self.declare_parameter('mask_topic',         '/lane/mask')
        self.declare_parameter('cmd_vel_topic',      '/cmd_vel')
        self.declare_parameter('counts_topic',       '/lane/segment_counts')
        self.declare_parameter('roi_top_frac',       0.5)
        self.declare_parameter('soft_threshold',     0.08)
        self.declare_parameter('hard_threshold',     0.15)
        self.declare_parameter('linear_speed',       0.15)
        self.declare_parameter('soft_angular_speed', 0.3)
        self.declare_parameter('hard_angular_speed', 0.7)

        # sliced into [0, 1, 2, 3, 4] w/ slice proportions
        self.declare_parameter('seg_frac_0', 0.25)   # outer left  (wide)
        self.declare_parameter('seg_frac_1', 0.15)   # inner left
        self.declare_parameter('seg_frac_2', 0.20)   # centre
        self.declare_parameter('seg_frac_3', 0.15)   # inner right
        self.declare_parameter('seg_frac_4', 0.25)   # outer right (wide)

        self.mask_topic         = self.get_parameter('mask_topic').value
        self.cmd_vel_topic      = self.get_parameter('cmd_vel_topic').value
        self.counts_topic       = self.get_parameter('counts_topic').value
        self.roi_top_frac       = self.get_parameter('roi_top_frac').value
        self.soft_threshold     = self.get_parameter('soft_threshold').value
        self.hard_threshold     = self.get_parameter('hard_threshold').value
        self.linear_speed       = self.get_parameter('linear_speed').value
        self.soft_angular_speed = self.get_parameter('soft_angular_speed').value
        self.hard_angular_speed = self.get_parameter('hard_angular_speed').value

        self.seg_fracs = [
            self.get_parameter(f'seg_frac_{i}').value for i in range(5)
        ]

     # ROS I/O
        self.mask_sub = self.create_subscription(
            Image,
            self.mask_topic,
            self.mask_callback,    # detector call
            1                      # CNN pub
        )

        # twist pub
        self.cmd_pub = self.create_publisher(
            Twist, self.cmd_vel_topic, 1)

        #pixel count pub
        self.counts_pub = self.create_publisher(
            Int32MultiArray, self.counts_topic, 1)

        self.bridge = CvBridge()

        self.get_logger().info(
            f'Controller node ready\n'
            f'  Receiving CNN mask from  : {self.mask_topic}\n'
            f'  Publishing Twist on      : {self.cmd_vel_topic}\n'
            f'  Publishing counts on     : {self.counts_topic}\n'
            f'  Segment fracs            : {self.seg_fracs}\n'
            f'  Soft threshold           : {self.soft_threshold * 100:.0f}% of ROI pixels\n'
            f'  Hard threshold           : {self.hard_threshold * 100:.0f}% of ROI pixels'
        )

    # for when CNN gived new mask
    def mask_callback(self, msg: Image):

        # image conversion
        mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        img_h, img_w = mask.shape

        #only looking at lower half
        roi_top = int(img_h * self.roi_top_frac)
        roi     = mask[roi_top:, :]         
        roi_h   = roi.shape[0]
        total_roi_pixels = roi_h * img_w    

       #compute slice boundaries based on percentage of max pixel count
        boundaries = [0]
        for frac in self.seg_fracs:
            boundaries.append(boundaries[-1] + int(img_w * frac))
        boundaries[-1] = img_w 

        # pixel count
        counts = []
        for i in range(5):
            col_start = boundaries[i]
            col_end   = boundaries[i + 1]
            segment   = roi[:, col_start:col_end]
            count     = int(np.count_nonzero(segment))
            counts.append(count)

    
        # twist command init/pub
        twist, command_label = self._decide_steering(counts, total_roi_pixels)
        self.cmd_pub.publish(twist)

        counts_msg      = Int32MultiArray()
        counts_msg.data = counts
        self.counts_pub.publish(counts_msg)

        self.get_logger().debug(
            f'Segments [S0={counts[0]:5d} S1={counts[1]:5d} '
            f'S2={counts[2]:5d} S3={counts[3]:5d} S4={counts[4]:5d}] '
            f'→ {command_label}'
        )

    # ── Steering decision logic ────────────────────────────────────────────────
    def _decide_steering(
            self,
            counts: list[int],
            total_roi_pixels: int
    ) -> tuple[Twist, str]:
        
        #set thresh hold pased on % of total pixels
        soft_px = self.soft_threshold * total_roi_pixels
        hard_px = self.hard_threshold * total_roi_pixels

        s0, s1, s2, s3, s4 = counts

        twist = Twist()
        twist.linear.x = self.linear_speed
        #turn based on where mx pixel count is located
        if s0 > hard_px:
            
            twist.angular.z = self.hard_angular_speed
            label = HARD_LEFT

        elif s4 > hard_px:
            twist.angular.z = -self.hard_angular_speed
            label = HARD_RIGHT

        elif s1 > soft_px:
            twist.angular.z = self.soft_angular_speed
            label = SOFT_LEFT

        elif s3 > soft_px: 
            twist.angular.z = -self.soft_angular_speed
            label = SOFT_RIGHT

        else:
            twist.angular.z = 0.0
            label = STRAIGHT

        return twist, label

    def destroy_node(self):
        # Stop robot when done
        self.cmd_pub.publish(Twist())
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()