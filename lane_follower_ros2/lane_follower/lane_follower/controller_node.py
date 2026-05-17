#!/usr/bin/env python3
"""
controller_node.py
==================
Subscribes to /lane/centroid (geometry_msgs/PointStamped) and publishes
geometry_msgs/Twist on /cmd_vel to keep the vehicle centred in the lane.

Control strategy
----------------
  error   = normalised horizontal offset of lane centre from image centre
             range [-1.0, +1.0]  (negative = centre is left of vehicle)

  angular.z = -Kp * error  - Kd * d(error)/dt
              (negative because ROS convention: +z turns left)

  linear.x  = base_speed * (1 - |error| * speed_reduction_factor)
              (slow down on tight curves)

A watchdog timer stops the vehicle if no centroid is received for
`timeout_sec` seconds (e.g. lane markings lost).

Parameters (config/params.yaml):
  base_speed            (float, default 0.15)  m/s
  kp                    (float, default 0.8)   proportional gain
  kd                    (float, default 0.1)   derivative gain
  speed_reduction       (float, default 0.5)   fraction of speed lost at max error
  max_angular_speed     (float, default 1.0)   rad/s clamp
  timeout_sec           (float, default 1.0)   watchdog timeout
  centroid_topic        (str,   default "/lane/centroid")
  cmd_vel_topic         (str,   default "/cmd_vel")
"""

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PointStamped


class ControllerNode(Node):

    def __init__(self):
        super().__init__('controller_node')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('base_speed',        0.15)
        self.declare_parameter('kp',                0.8)
        self.declare_parameter('kd',                0.1)
        self.declare_parameter('speed_reduction',   0.5)
        self.declare_parameter('max_angular_speed', 1.0)
        self.declare_parameter('timeout_sec',       1.0)
        self.declare_parameter('centroid_topic',    '/lane/centroid')
        self.declare_parameter('cmd_vel_topic',     '/cmd_vel')

        self.base_speed     = self.get_parameter('base_speed').value
        self.kp             = self.get_parameter('kp').value
        self.kd             = self.get_parameter('kd').value
        self.speed_reduce   = self.get_parameter('speed_reduction').value
        self.max_ang        = self.get_parameter('max_angular_speed').value
        timeout             = self.get_parameter('timeout_sec').value
        cent_topic          = self.get_parameter('centroid_topic').value
        cmd_topic           = self.get_parameter('cmd_vel_topic').value

        # ── State ─────────────────────────────────────────────────────────────
        self.prev_error     = 0.0
        self.prev_time      = self.get_clock().now()
        self.last_centroid  = None   # set by callback; cleared by watchdog

        # ── ROS I/O ───────────────────────────────────────────────────────────
        self.cmd_pub  = self.create_publisher(Twist, cmd_topic, 10)
        self.sub      = self.create_subscription(
            PointStamped, cent_topic, self.centroid_callback, 10
        )
        # Publish at 20 Hz regardless of centroid rate
        self.control_timer  = self.create_timer(0.05, self.control_loop)
        # Watchdog: zero speed if no centroid for timeout_sec
        self.watchdog_timer = self.create_timer(timeout, self.watchdog)

        self.get_logger().info(
            f'Controller ready | Kp={self.kp} Kd={self.kd} '
            f'base_speed={self.base_speed} m/s'
        )

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def centroid_callback(self, msg: PointStamped):
        """Cache the latest centroid. z carries image width for normalisation."""
        self.last_centroid = msg
        self.watchdog_timer.reset()   # reset watchdog on every good detection

    def watchdog(self):
        """Called only when no centroid arrives within timeout_sec — stop vehicle."""
        self.get_logger().warn(
            'Lane centroid timeout — stopping vehicle for safety'
        )
        self.last_centroid = None
        self._publish_twist(0.0, 0.0)

    def control_loop(self):
        if self.last_centroid is None:
            return   # watchdog already handles the stop case

        cx        = self.last_centroid.point.x
        img_width = self.last_centroid.point.z   # passed through as image width
        if img_width == 0:
            return

        # ── PD control ────────────────────────────────────────────────────────
        # Normalise cx to [-1, +1] where 0 = image centre
        error = (cx - img_width / 2.0) / (img_width / 2.0)

        now      = self.get_clock().now()
        dt       = (now - self.prev_time).nanoseconds * 1e-9
        dt       = max(dt, 1e-4)   # guard against div-by-zero
        d_error  = (error - self.prev_error) / dt

        angular  = -(self.kp * error + self.kd * d_error)
        angular  = float(max(-self.max_ang, min(self.max_ang, angular)))

        # Slow down proportionally to steering effort
        linear   = self.base_speed * (1.0 - self.speed_reduce * abs(error))
        linear   = float(max(0.0, linear))

        self.prev_error = error
        self.prev_time  = now

        self._publish_twist(linear, angular)

        self.get_logger().debug(
            f'error={error:+.3f}  angular={angular:+.3f}  linear={linear:.3f}'
        )

    def _publish_twist(self, linear: float, angular: float):
        msg                  = Twist()
        msg.linear.x         = linear
        msg.angular.z        = angular
        self.cmd_pub.publish(msg)

    def destroy_node(self):
        """Ensure vehicle stops when node shuts down."""
        self._publish_twist(0.0, 0.0)
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
