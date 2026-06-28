#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, Range
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String

import cv2
import numpy as np
from cv_bridge import CvBridge
import math


# ─── HSV colour bounds ───────────────────────────────────────────────────────

RED_LOWER1 = np.array([0,   120,  70])
RED_UPPER1 = np.array([10,  255, 255])
RED_LOWER2 = np.array([170, 120,  70])
RED_UPPER2 = np.array([180, 255, 255])

GREEN_LOWER = np.array([40,  80,  80])
GREEN_UPPER = np.array([80, 255, 255])

OBSTACLE_LOWER = np.array([0,   0,   0])
OBSTACLE_UPPER = np.array([180, 255,  80])

# ─── Navigation states ────────────────────────────────────────────────────────

STATE_TAKEOFF       = 'TAKEOFF'
STATE_FIND_GOAL     = 'FIND_GOAL'
STATE_AVOID_LEFT    = 'AVOID_LEFT'
STATE_AVOID_RIGHT   = 'AVOID_RIGHT'
STATE_AVOID_BACK    = 'AVOID_BACK'
STATE_ROTATE        = 'ROTATE'
STATE_HOVER         = 'HOVER'
STATE_MISSION_DONE  = 'MISSION_DONE'
STATE_ALIGN_GOAL    = 'ALIGN_GOAL'

# ─── Obstacle zone thresholds (fraction of image width) ───────────────────────

ZONE_LEFT_MAX   = 0.35
ZONE_RIGHT_MIN  = 0.65
ZONE_CENTER_MIN = 0.35
ZONE_CENTER_MAX = 0.65

OBSTACLE_MIN_AREA = 3000

GOAL_MIN_AREA      = 1500
GOAL_CENTER_TOL    = 0.15
GOAL_AREA_THRESHOLD = 8000   # area required for mission completion
GOAL_LOCK_FRAMES    = 15     # consecutive aligned frames before MISSION_COMPLETE
ALIGN_SPEED         = 0.07   # m/s for fine lateral / fore-aft alignment
ALIGN_TRIGGER_AREA  = 3000   # minimum goal area to enter STATE_ALIGN_GOAL

TARGET_ALT   = 2.0
ALT_KP       = 0.8
ALT_DEADBAND = 0.08

FWD_SPEED  = 0.35
SIDE_SPEED = 0.40
YAW_SPEED  = 0.35
VERT_SPEED = 0.50


class DroneNavigator(Node):
    """Camera-based obstacle avoidance and goal seeking for a UAV."""

    def __init__(self):
        super().__init__('drone_navigator')

        self.bridge = CvBridge()

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ── Subscriber 1: front camera — obstacle detection ONLY ──────────────
        self.create_subscription(
            Image, '/drone/front_camera/image_raw',
            self.front_camera_callback, qos_sensor)

        # ── Subscriber 2: down camera — goal detection ONLY ───────────────────
        self.create_subscription(
            Image, '/drone/down_camera/image_raw',
            self.down_camera_callback, qos_sensor)

        self.create_subscription(
            Odometry, '/drone/odom',
            self.odom_callback, 10)

        # ── Publishers ────────────────────────────────────────────────────────
        self.cmd_pub    = self.create_publisher(Twist,  '/drone/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/drone/mission_status', 10)

        # ── State ─────────────────────────────────────────────────────────────
        self.state       = STATE_TAKEOFF
        self.altitude    = 0.0
        self.current_x   = 0.0
        self.current_y   = 0.0
        self.current_yaw = 0.0

        # Obstacle state (set only by front camera)
        self.obstacle_left   = False
        self.obstacle_center = False
        self.obstacle_right  = False

        # Goal state (set only by down camera)
        self.goal_visible    = False
        self.goal_centered   = False
        self.goal_area       = 0.0
        self.goal_cx         = 0
        self.goal_cy         = 0

        # Front camera frame dimensions
        self.front_img_width  = 640
        self.front_img_height = 480

        # Down camera frame dimensions
        self.down_img_width  = 640
        self.down_img_height = 480

        self.avoid_timer  = 0
        self.rotate_dir   = 1

        self.takeoff_ticks    = 0
        self.mission_complete = False
        self.goal_lock_counter = 0   # consecutive aligned frames counter

        # Control loop at 20 Hz
        self.create_timer(0.05, self.control_loop)

        self.get_logger().info('DroneNavigator initialised — dual camera pipeline active.')

    # ─────────────────────────────────────────────────────────────────────────
    # Callbacks
    # ─────────────────────────────────────────────────────────────────────────

    def odom_callback(self, msg: Odometry):
        self.altitude  = msg.pose.pose.position.z
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def front_camera_callback(self, msg: Image):
        """Front camera — obstacle detection ONLY. Never touches goal state."""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'front cv_bridge error: {e}')
            return

        self.front_img_height, self.front_img_width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        self._detect_obstacles(frame, hsv)
        self._draw_front_debug(frame)

    def down_camera_callback(self, msg: Image):
        """Down camera — goal detection ONLY. Never touches obstacle state."""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'down cv_bridge error: {e}')
            return

        self.down_img_height, self.down_img_width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        self._detect_goal(frame, hsv)
        self._draw_down_debug(frame)

    # ─────────────────────────────────────────────────────────────────────────
    # Perception — obstacle detection (front camera only)
    # ─────────────────────────────────────────────────────────────────────────

    def _suppress_red(self, frame, hsv):
        """Return frame copy with red pixels set to mid-grey.
        Prevents the goal marker from triggering obstacle detection."""
        mask1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
        mask2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        red_mask = cv2.dilate(red_mask, np.ones((7, 7), np.uint8), iterations=1)
        suppressed = frame.copy()
        suppressed[red_mask > 0] = (128, 128, 128)
        return suppressed

    def _detect_obstacles(self, frame, hsv):
        """
        Detect physical 3-D objects (chairs) using edge/contour analysis.
        Red pixels are suppressed first so the goal marker is never detected
        as an obstacle. No goal-camera variables are touched here.
        """
        # 1. Suppress red so goal marker is invisible to this pipeline
        clean   = self._suppress_red(frame, hsv)

        # 2. Grayscale → Gaussian blur → Canny
        gray    = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges   = cv2.Canny(blurred, 40, 120)

        # 3. Morphological close to join broken chair edges
        kernel = np.ones((7, 7), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # 4. ROI: ignore sky (top 15 %) and floor (bottom 20 %)
        roi_top    = int(self.front_img_height * 0.15)
        roi_bottom = int(self.front_img_height * 0.80)
        obstacle_roi = closed[roi_top:roi_bottom, :]

        # 5. Find and filter contours
        contours, _ = cv2.findContours(
            obstacle_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        MIN_BB_WIDTH  = 20    # px — rejects thin lines
        MIN_BB_HEIGHT = 30    # px — rejects flat floor markings
        MIN_ASPECT    = 0.20  # bh/bw — rejects very horizontal shapes
        MIN_AREA      = OBSTACLE_MIN_AREA

        w = self.front_img_width
        left_area   = 0.0
        center_area = 0.0
        right_area  = 0.0
        valid_contours = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < MIN_BB_WIDTH or bh < MIN_BB_HEIGHT:
                continue
            if (bh / float(bw)) < MIN_ASPECT:
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue

            cx = int(M['m10'] / M['m00'])
            norm_x = cx / w
            valid_contours.append(cnt)

            if norm_x < ZONE_LEFT_MAX:
                left_area += area
            elif norm_x > ZONE_RIGHT_MIN:
                right_area += area
            else:
                center_area += area

        self.obstacle_left   = left_area   > MIN_AREA
        self.obstacle_center = center_area > MIN_AREA
        self.obstacle_right  = right_area  > MIN_AREA

        # Per-column obstacle pixel count — used by _obstacle_action()
        # to pick the freer side when only CENTER is blocked
        self._obs_col_density = obstacle_roi.sum(axis=0) / 255.0  # shape (W,)

        self._last_obs_mask     = closed
        self._last_obs_contours = valid_contours
        self._obs_roi_top       = roi_top

    # ─────────────────────────────────────────────────────────────────────────
    # Perception — goal detection (down camera only)
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_goal(self, frame, hsv):
        mask1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
        mask2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN,
                                    np.ones((5, 5), np.uint8))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE,
                                    np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(
            red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.goal_visible  = False
        self.goal_centered = False
        self.goal_area     = 0.0
        self.goal_cx       = self.down_img_width // 2
        self._last_goal_contour = None

        if not contours:
            return

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < GOAL_MIN_AREA:
            return

        M = cv2.moments(largest)
        if M['m00'] == 0:
            return

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        self.goal_cx      = cx
        self.goal_cy      = cy
        self.goal_area    = area
        self.goal_visible = True
        self._last_goal_contour = largest

        half_w = self.down_img_width / 2.0
        offset = abs(cx - half_w) / half_w
        self.goal_centered = offset < GOAL_CENTER_TOL

    # ─────────────────────────────────────────────────────────────────────────
    # Debug visualisation — Window 1: Front Camera - Obstacle Detection
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_front_debug(self, frame):
        debug = frame.copy()
        h, w = debug.shape[:2]

        # Zone divider lines
        cv2.line(debug, (int(w * ZONE_LEFT_MAX), 0),
                 (int(w * ZONE_LEFT_MAX), h), (255, 255, 0), 2)
        cv2.line(debug, (int(w * ZONE_RIGHT_MIN), 0),
                 (int(w * ZONE_RIGHT_MIN), h), (255, 255, 0), 2)

        # Zone labels
        cv2.putText(debug, 'LEFT', (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(debug, 'CENTER', (int(w * ZONE_LEFT_MAX) + 10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(debug, 'RIGHT', (int(w * ZONE_RIGHT_MIN) + 10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Zone highlight when obstacle detected
        overlay = debug.copy()
        if self.obstacle_left:
            cv2.rectangle(overlay, (0, 0),
                          (int(w * ZONE_LEFT_MAX), h), (0, 0, 255), -1)
            cv2.putText(debug, 'OBSTACLE', (5, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        if self.obstacle_center:
            cv2.rectangle(overlay,
                          (int(w * ZONE_LEFT_MAX), 0),
                          (int(w * ZONE_RIGHT_MIN), h), (0, 100, 255), -1)
            cv2.putText(debug, 'OBSTACLE',
                        (int(w * ZONE_LEFT_MAX) + 5, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 2)
        if self.obstacle_right:
            cv2.rectangle(overlay,
                          (int(w * ZONE_RIGHT_MIN), 0), (w, h),
                          (0, 0, 255), -1)
            cv2.putText(debug, 'OBSTACLE',
                        (int(w * ZONE_RIGHT_MIN) + 5, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        cv2.addWeighted(overlay, 0.25, debug, 0.75, 0, debug)

        # Obstacle bounding boxes from contours
        if hasattr(self, '_last_obs_contours'):
            roi_top = getattr(self, '_obs_roi_top', 0)
            for cnt in self._last_obs_contours:
                if cv2.contourArea(cnt) < OBSTACLE_MIN_AREA:
                    continue
                x, y, bw, bh = cv2.boundingRect(cnt)
                cv2.rectangle(debug,
                              (x, y + roi_top),
                              (x + bw, y + roi_top + bh),
                              (0, 255, 255), 2)

        # Determine avoidance decision label
        L = self.obstacle_left
        C = self.obstacle_center
        R = self.obstacle_right
        if not L and not C and not R:
            decision = 'CLEAR - FORWARD'
            dcol = (0, 255, 0)
        elif L and R and C:
            decision = 'FULL BLOCK - ROTATE'
            dcol = (0, 0, 255)
        elif L and C and not R:
            decision = 'AVOID RIGHT'
            dcol = (0, 165, 255)
        elif R and C and not L:
            decision = 'AVOID LEFT'
            dcol = (0, 165, 255)
        elif C:
            decision = 'CENTER BLOCK - CHOOSE SIDE'
            dcol = (0, 165, 255)
        elif L:
            decision = 'LEFT BLOCK - MOVE RIGHT'
            dcol = (0, 200, 255)
        elif R:
            decision = 'RIGHT BLOCK - MOVE LEFT'
            dcol = (0, 200, 255)
        else:
            decision = 'CLEAR'
            dcol = (0, 255, 0)

        # HUD
        cv2.putText(debug, f'State: {self.state}',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(debug, f'Ticks: {self.takeoff_ticks}',
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        cv2.putText(debug, f'Decision: {decision}',
                    (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, dcol, 2)

        cv2.imshow('Front Camera - Obstacle Detection', debug)
        cv2.waitKey(1)

    # ─────────────────────────────────────────────────────────────────────────
    # Debug visualisation — Window 2: Down Camera - Goal Detection
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_down_debug(self, frame):
        debug = frame.copy()
        h, w = debug.shape[:2]

        if self.goal_visible and hasattr(self, '_last_goal_contour') \
                and self._last_goal_contour is not None:
            # Bounding box around goal
            x, y, bw, bh = cv2.boundingRect(self._last_goal_contour)
            cv2.rectangle(debug, (x, y), (x + bw, y + bh), (0, 0, 255), 3)

            # Goal centre crosshair
            cy_pos = y + bh // 2
            cv2.line(debug, (self.goal_cx - 20, cy_pos),
                     (self.goal_cx + 20, cy_pos), (0, 0, 255), 2)
            cv2.line(debug, (self.goal_cx, cy_pos - 20),
                     (self.goal_cx, cy_pos + 20), (0, 0, 255), 2)
            cv2.circle(debug, (self.goal_cx, cy_pos), 30, (0, 0, 255), 2)

            cv2.putText(debug, f'Area: {self.goal_area:.0f}',
                        (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)
            cv2.putText(debug, f'cx: {self.goal_cx}',
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)

        # Image centre reference line
        cv2.line(debug, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)

        # Status
        goal_str   = 'VISIBLE' if self.goal_visible else 'NOT FOUND'
        center_str = 'CENTRED' if self.goal_centered else 'OFF-CENTRE'
        gcol = (0, 255, 0) if self.goal_visible else (100, 100, 100)

        cv2.putText(debug, f'Goal: {goal_str}',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, gcol, 2)
        cv2.putText(debug, f'Align: {center_str}',
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, gcol, 2)
        cv2.putText(debug, f'Area: {self.goal_area:.0f} / 8000',
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(debug, f'Center X: {self.goal_cx}  ImgW: {self.down_img_width}',
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        cv2.putText(debug, f'State: {self.state}',
                    (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

        if self.mission_complete:
            overlay = debug.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.35, debug, 0.65, 0, debug)
            cv2.putText(debug, 'MISSION COMPLETE',
                        (w // 2 - 140, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow('Down Camera - Goal Detection', debug)
        cv2.waitKey(1)

    # ─────────────────────────────────────────────────────────────────────────
    # Altitude hold
    # ─────────────────────────────────────────────────────────────────────────

    def _altitude_correction(self) -> float:
        error = TARGET_ALT - self.altitude
        if abs(error) < ALT_DEADBAND:
            return 0.0
        return float(np.clip(ALT_KP * error, -VERT_SPEED, VERT_SPEED))

    # ─────────────────────────────────────────────────────────────────────────
    # Obstacle action (unchanged logic)
    # ─────────────────────────────────────────────────────────────────────────

    def _free_space_side(self) -> str:
        """
        Compare obstacle pixel density in the left half vs right half of the
        front-camera ROI and return 'LEFT' or 'RIGHT' for the freer side.
        Falls back to 'LEFT' when column density is unavailable.
        No goal-camera variables are used.
        """
        if not hasattr(self, '_obs_col_density') or self._obs_col_density.size == 0:
            return 'LEFT'
        mid   = len(self._obs_col_density) // 2
        left_density  = float(self._obs_col_density[:mid].sum())
        right_density = float(self._obs_col_density[mid:].sum())
        return 'LEFT' if left_density <= right_density else 'RIGHT'

    def _obstacle_action(self):
        """
        Pure front-camera obstacle avoidance.
        Does NOT reference goal_visible, goal_cx, goal_area, or any
        variable produced by the down camera.
        """
        L = self.obstacle_left
        C = self.obstacle_center
        R = self.obstacle_right

        if not L and not C and not R:
            return 'CLEAR', FWD_SPEED, 0.0, 0.0

        if L and R and C:
            return 'FULL_BLOCK', 0.0, 0.0, 0.0

        if L and C and not R:
            return 'AVOID_RIGHT', 0.1, -SIDE_SPEED, 0.0

        if R and C and not L:
            return 'AVOID_LEFT', 0.1, SIDE_SPEED, 0.0

        if C and not L and not R:
            # Choose the side with fewer obstacle pixels — no goal variables
            if self._free_space_side() == 'LEFT':
                return 'AVOID_LEFT',  0.1,  SIDE_SPEED, 0.0
            else:
                return 'AVOID_RIGHT', 0.1, -SIDE_SPEED, 0.0

        if L and not C and not R:
            return 'AVOID_RIGHT', FWD_SPEED * 0.7, -SIDE_SPEED * 0.6, 0.0

        if R and not C and not L:
            return 'AVOID_LEFT', FWD_SPEED * 0.7, SIDE_SPEED * 0.6, 0.0

        return 'FULL_BLOCK', 0.0, 0.0, YAW_SPEED * self.rotate_dir

    # ─────────────────────────────────────────────────────────────────────────
    # Goal-seeking yaw correction (uses down camera goal_cx)
    # ─────────────────────────────────────────────────────────────────────────

    def _goal_yaw_correction(self) -> float:
        if not self.goal_visible:
            return 0.0
        half  = self.down_img_width / 2.0
        error = (self.goal_cx - half) / half
        kp    = 0.5
        return float(np.clip(-kp * error, -YAW_SPEED, YAW_SPEED))

    # ─────────────────────────────────────────────────────────────────────────
    # Goal alignment (down camera only — no yaw, no altitude change)
    # ─────────────────────────────────────────────────────────────────────────

    def _align_over_goal(self) -> Twist:
        """
        Return a Twist that nudges the drone laterally / fore-aft to centre
        the goal in the DOWN camera frame.
        No rotation.  No altitude change.  Very small velocities.
        Uses ONLY self.goal_cx / goal_cy (down camera).
        """
        cmd = Twist()

        if not self.goal_visible:
            return cmd

        half_w = self.down_img_width  / 2.0
        half_h = self.down_img_height / 2.0

        x_err = (self.goal_cx - half_w) / half_w   # −1…+1 (left<0, right>0)
        y_err = (self.goal_cy - half_h) / half_h   # −1…+1 (above<0, below>0)

        # Left / right in image → drone body Y axis
        if x_err < -0.05:           # goal is LEFT  → move left
            cmd.linear.y =  ALIGN_SPEED
        elif x_err > 0.05:          # goal is RIGHT → move right
            cmd.linear.y = -ALIGN_SPEED

        # Above / below in image → drone body X axis
        if y_err < -0.05:           # goal is ABOVE centre → move forward
            cmd.linear.x =  ALIGN_SPEED
        elif y_err > 0.05:          # goal is BELOW centre → move backward
            cmd.linear.x = -ALIGN_SPEED

        cmd.linear.z  = 0.0
        cmd.angular.z = 0.0
        return cmd

    # ─────────────────────────────────────────────────────────────────────────
    # Main control loop (unchanged logic)
    # ─────────────────────────────────────────────────────────────────────────

    def control_loop(self):
        cmd = Twist()

        # FIX 3: publish state every loop tick
        status = String()
        status.data = (
            f'{self.state}'
            f'|obs_L:{self.obstacle_left}'
            f'|C:{self.obstacle_center}'
            f'|R:{self.obstacle_right}'
            f'|goal:{self.goal_visible}'
        )
        self.status_pub.publish(status)

        if self.mission_complete:
            self.cmd_pub.publish(cmd)
            return

        # FIX 2: vz is always 0.0 after takeoff — no altitude correction
        vz = 0.0

        # FIX 1: timer-based takeoff — 60 ticks at 20 Hz = 3 seconds
        if self.state == STATE_TAKEOFF:
            cmd.linear.z = VERT_SPEED
            self.takeoff_ticks += 1
            if self.takeoff_ticks >= 60:
                self.state = STATE_FIND_GOAL
                self.get_logger().info('Takeoff complete (3 s) → FIND_GOAL')
            self.cmd_pub.publish(cmd)
            return

        if self.state == STATE_MISSION_DONE:
            cmd.linear.z = 0.0
            self.cmd_pub.publish(cmd)
            return

        # ── Enter alignment phase only when goal is close enough ─────────────
        if (self.goal_visible
                and self.goal_area > ALIGN_TRIGGER_AREA
                and self.state != STATE_MISSION_DONE):
            self.state = STATE_ALIGN_GOAL

        # ── STATE_ALIGN_GOAL — fine-position over goal before completing ──────
        if self.state == STATE_ALIGN_GOAL:

            # FIX 3: goal lost during alignment → resume searching
            if not self.goal_visible:
                self.goal_lock_counter = 0
                self.state = STATE_FIND_GOAL
                # fall through to normal navigation below
            else:
                cmd = self._align_over_goal()

                all_good = (
                    self.goal_visible
                    and self.goal_centered
                    and self.goal_area > GOAL_AREA_THRESHOLD
                )

                if all_good:
                    self.goal_lock_counter += 1
                else:
                    self.goal_lock_counter = 0   # reset on any drift

                if self.goal_lock_counter >= GOAL_LOCK_FRAMES:
                    self.state            = STATE_MISSION_DONE
                    self.mission_complete = True
                    self.get_logger().info(
                        f'🎯 Goal locked for {GOAL_LOCK_FRAMES} frames — MISSION COMPLETE')
                    status.data = f'{STATE_MISSION_DONE}|MISSION_COMPLETE'
                    self.status_pub.publish(status)
                    self.cmd_pub.publish(Twist())   # zero velocity — hover in place
                    return

                self.cmd_pub.publish(cmd)
                return

        action, vx, vy, wz = self._obstacle_action()

        if action == 'CLEAR':
            cmd.linear.x  = FWD_SPEED
            cmd.linear.y  = vy
            cmd.linear.z  = 0.0
            cmd.angular.z = 0.0
            self.state    = STATE_FIND_GOAL

        elif action == 'FULL_BLOCK':
            self.avoid_timer += 1
            if self.avoid_timer > 30:
                self.rotate_dir  = -self.rotate_dir
                self.avoid_timer = 0
            cmd.linear.x  = -0.1
            cmd.linear.z  = 0.0
            cmd.angular.z = YAW_SPEED * self.rotate_dir
            self.state    = STATE_ROTATE

        else:
            cmd.linear.x  = float(vx)
            cmd.linear.y  = float(vy)
            cmd.linear.z  = 0.0
            cmd.angular.z = float(wz)
            self.state    = STATE_FIND_GOAL

        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = DroneNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()