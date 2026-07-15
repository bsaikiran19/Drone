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
import heapq          # A* open-list (min-heap)


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

# ─── V2.2 Part 2: Precision Alignment (landing-pad pixel-error centring) ──────
STATE_PRECISION_ALIGN = 'PRECISION_ALIGN'

# ─── V2.3 Part 1: Controlled Descent (vertical descent + horizontal hold) ─────
STATE_CONTROLLED_DESCENT = 'CONTROLLED_DESCENT'

# ─── V2.3 Part 2: Final Soft Landing & Mission Completion ─────────────────────
STATE_FINAL_LANDING    = 'FINAL_LANDING'
STATE_MISSION_COMPLETE = 'MISSION_COMPLETE'

# ─── V2: Path planning states (reserved — not active yet) ────────────────────

STATE_PLAN_PATH      = 'PLAN_PATH'       # compute or receive a waypoint path
STATE_FOLLOW_PATH    = 'FOLLOW_PATH'     # execute waypoints in sequence
STATE_RETURN_TO_PATH = 'RETURN_TO_PATH'  # re-join path after obstacle detour

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

# ─── V2.2 Part 2: Precision Alignment constants ───────────────────────────────
PRECISION_PIXEL_TOL    = 20     # px — tolerance window on each axis (20 px avoids oscillation)
PRECISION_MAX_SPEED    = 0.12   # m/s — hard cap on P-controller output (V2.5b, was 0.15 — reduced, was saturating)
PRECISION_KP           = 0.0015 # V2.5b — P-controller gain (was 0.0025 — reduced, was saturating almost immediately)
PRECISION_DEADBAND_PX  = 5      # px — inside this band the axis is commanded 0.0

# ─── V2.7: alignment timeout / pad-lost grace (auto axis-flip REMOVED) ────────
CONTROL_LOOP_HZ                = 20.0   # matches create_timer(0.05, control_loop)
ALIGN_LOCK_FRAMES              = 10     # consecutive aligned frames → STATE_CONTROLLED_DESCENT
PAD_LOST_GRACE_SEC             = 0.5    # s — reuse the last good correction before stopping
PAD_LOST_GRACE_TICKS           = int(PAD_LOST_GRACE_SEC * CONTROL_LOOP_HZ)

# ─── V2.3 Part 1: Controlled Descent constants ────────────────────────────────
DESCENT_SPEED    = -0.10  # m/s — speed while actively moving during a coarse descent step
DESCENT_ALTITUDE = 0.35   # m   — stop controlled descent once altitude reaches this

# ─── V2.3 Part 2: Final Soft Landing constants ────────────────────────────────
FINAL_LANDING_ALTITUDE = 0.12   # m — altitude at which the final, slowest descent rate kicks in
TOUCHDOWN_ALTITUDE     = 0.03   # m — altitude that signals touchdown

# ─── V2.7: Incremental (stepped) descent — replaces continuous descent ────────
# Uses ONLY PRECISION_PIXEL_TOL for alignment gating everywhere (DESCENT_ALIGN_TOL
# removed — no looser tolerance is ever used during landing).
DESCENT_STEP_METERS       = 0.05   # m   — descend ~5 cm, stop, re-check alignment, repeat
FINAL_DESCENT_STEP_SPEED  = -0.05  # m/s — slower step speed used during STATE_FINAL_LANDING
ALIGN_STABLE_FRAMES       = 6      # consecutive aligned frames required before resuming
                                    # a paused descent, and before starting each next step

# ─── V2.6: Multi-condition touchdown detection ────────────────────────────────
TOUCHDOWN_VZ_TOL        = 0.02   # m/s   — |vertical velocity| must be under this
TOUCHDOWN_XY_TOL        = 0.03   # m     — horizontal drift per tick must be under this
TOUCHDOWN_CONFIRM_TICKS = 8      # consecutive ticks all conditions must hold (0.4 s)
MOTOR_DISABLE_HOLD_TICKS = 20    # ticks of confirmed rest after touchdown before disabling motors (~1.0 s)

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
        self.enable_pub = self.create_publisher(Bool,   '/drone/enable',  10)
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

        # ── V2.2: Precision Landing — detection-only state (down camera) ───────
        # Populated by _detect_landing_pad(); never written to by navigation.
        self.landing_pad_found  = False
        self.landing_pad_cx     = 0
        self.landing_pad_cy     = 0
        self.landing_pad_radius = 0
        self.landing_pad_area   = 0.0
        self.landing_pixel_err_x = 0
        self.landing_pixel_err_y = 0

        # ── V2.2 Part 2: print PRECISION ALIGNMENT COMPLETE only once per lock ──
        self.alignment_counter            = 0   # consecutive frames within ±20 px; threshold = 10

        # ── V2.5: P-controller telemetry (debug window only, no logic impact) ──
        self._align_vx           = 0.0   # last commanded body X velocity
        self._align_vy           = 0.0   # last commanded body Y velocity
        self._precision_p_out_x  = 0.0   # raw P output, X-image axis
        self._precision_p_out_y  = 0.0   # raw P output, Y-image axis

        # ── V2.5 — Problem 4: odom freshness flag. Fixes the debug window
        # showing "Altitude: 0.000" before the first /drone/odom message
        # arrives. No new subscriber — reuses the existing odom_callback.
        self._odom_received = False

        # ── V2.7: automatic axis-flip system REMOVED per V2.4 Final Stable spec.
        # precision_align() now uses one fixed coordinate mapping only.

        # ── V2.5b Problem 5: pad-lost grace period state ─────────────────────────
        self._align_pad_lost_ticks = 0        # consecutive ticks with no pad detected
        self._last_precision_cmd   = Twist()  # last good correction, reused during grace

        # ── V2.5b: pad-lost grace state ──────────────────────────────────────────

        # ── V2.3 Part 1: Controlled Descent state (display + hover-limit) ──────
        self.descent_active       = False   # True while cmd.linear.z = DESCENT_SPEED
        self.descent_hover_ready  = False   # True once altitude threshold reached

        # ── V2.3 Part 2: Final Soft Landing / Mission Complete state ───────────
        self.touchdown_occurred    = False   # True once multi-condition touchdown confirms
        self.mission_complete_logged = False # ensures MISSION SUCCESSFUL prints once
        self.landing_timer_ticks   = 0      # ticks counted after touchdown, until motor disable

        # ── V2.7: alignment-gated INCREMENTAL descent state ─────────────────────
        self.descent_paused                = False  # True while vertical motion is paused for realignment
        self.descent_realign_lock          = 0       # consecutive aligned frames while paused/between steps
        self.descent_step_active           = False   # True while actively moving through one 5 cm step
        self.descent_step_target_altitude  = None    # altitude at which the current step ends

        # ── V2.6: multi-condition touchdown state ───────────────────────────────
        self._vertical_velocity      = 0.0
        self._prev_altitude_for_vz   = 0.0
        self._horizontal_drift       = 0.0
        self._prev_x_for_stability   = 0.0
        self._prev_y_for_stability   = 0.0
        self.touchdown_confirm_ticks = 0

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

        # ── V2: Path planning variables (unused until A* is implemented) ───────
        self.path               = []    # list of (x, y) waypoints in world frame
        self.current_waypoint   = 0     # index into self.path
        self.path_ready         = False # True once plan_path() produces a valid path
        self.returning_to_path  = False # True while drone re-joins path after detour

        # ── Occupancy Grid (V2 Phase 1 — infrastructure only) ─────────────────
        # Grid covers the full arena: X [0, 10] m, Y [-5, 5] m
        # Cell size 0.5 m → 20 columns (X) × 20 rows (Y)
        self.grid_cols       = 20          # number of cells along world X axis
        self.grid_rows       = 20          # number of cells along world Y axis
        self.grid_origin_x   = 0.0         # world X that maps to grid column 0
        self.grid_origin_y   = -5.0        # world Y that maps to grid row 0
        self.grid_cell_size  = 0.5         # metres per cell (both axes)

        # Inflate obstacles by this many cells so A* paths clear chair edges.
        # Set to 1 for a single-cell safety margin (matches chair ~0.5 m footprint).
        self.grid_inflation  = 1

        # Static obstacle list — (world_x, world_y) of each static chair centre.
        # Source: drone_arena.sdf, models with <static>true</static>.
        # Moving chairs (chair_1–4, no <static> tag) are intentionally excluded
        # because they are handled by the existing OpenCV local avoidance system.
        self.static_obstacles = [
            (1.5,  0.9),   # chair_5  — Zone A upper entry blocker
            (1.5, -0.9),   # chair_6  — Zone A lower entry blocker
            (2.5,  1.8),   # chair_7  — Zone B upper corridor wall
            (3.5,  1.8),   # chair_8  — Zone B upper corridor wall
            (2.5, -2.2),   # chair_9  — Zone C lower corridor wall
            (3.5, -2.2),   # chair_10 — Zone C lower corridor wall
            (4.5,  0.0),   # chair_11 — Zone D centre-line pinch
            (5.5,  2.8),   # chair_12 — Zone D dead-end pocket back wall
            (4.8,  2.2),   # chair_13 — Zone D dead-end pocket left side wall
            (5.5, -2.5),   # chair_14 — Zone D open-space scatter
            (6.0, -1.8),   # chair_15 — open-space scatter
            (7.5,  1.2),   # chair_16 — Zone E upper gate post
            (7.5, -1.2),   # chair_17 — Zone E lower gate post
            (8.0,  1.8),   # chair_18 — approach scatter upper
            (8.0, -1.8),   # chair_19 — approach scatter lower
            (6.5, -0.9),   # chair_20 — mid-arena open-space filler
        ]

        # Build and store the grid once at startup.
        # No navigation behaviour is affected; the grid is passive data.
        self.occupancy_grid = self.build_occupancy_grid()
        self.print_grid_debug()

        # ── V2.2: waypoint controller smoothing state (previous published vx/vy) ─
        self._wp_prev_vx = 0.0
        self._wp_prev_vy = 0.0

        # ── V2.1: A* Planner Visualization (display-only, no logic impact) ────
        self._astar_explored_cells = []   # cells expanded during the last A* run
        self._astar_grid_path      = []   # final (col, row) path from the last run
        self._astar_nodes_expanded = 0
        self._astar_planning_ms    = 0.0
        self._viz_cell_px          = 20   # pixels per grid cell
        # Visualization refresh at ~10 FPS — independent of the 20 Hz control loop
        self.create_timer(0.1, self.draw_astar_visualization)

        # Landing debug log at 1 Hz — Step 7 (does not spam the terminal)
        self.create_timer(1.0, self._landing_debug_log)

        # Control loop at 20 Hz
        self.create_timer(0.05, self.control_loop)

        self.get_logger().info('=' * 50)
        self.get_logger().info('MISSION STARTED')
        self.get_logger().info('=' * 50)

    # ─────────────────────────────────────────────────────────────────────────
    # Callbacks
    # ─────────────────────────────────────────────────────────────────────────

    def _update_landing_telemetry(self):
        """
        V2.6 — Runs once per control-loop tick regardless of state. Maintains
        a simple finite-difference vertical velocity and horizontal drift
        estimate, used only by the landing controller for multi-condition
        touchdown detection. Never affects any non-landing state or command.
        """
        dt = 1.0 / CONTROL_LOOP_HZ
        self._vertical_velocity = (self.altitude - self._prev_altitude_for_vz) / dt
        self._prev_altitude_for_vz = self.altitude

        self._horizontal_drift = math.hypot(
            self.current_x - self._prev_x_for_stability,
            self.current_y - self._prev_y_for_stability,
        )
        self._prev_x_for_stability = self.current_x
        self._prev_y_for_stability = self.current_y

    def odom_callback(self, msg: Odometry):
        self.altitude  = msg.pose.pose.position.z
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

        # V2.5 — Problem 4: self.altitude is already correct; this flag only
        # lets the debug window distinguish "no odom yet" from "altitude 0".
        self._odom_received = True

    def _landing_debug_log(self):
        """
        V2.6 — 1 Hz terminal debug: Current State, Altitude, measured
        Vertical Velocity, Pixel Error X/Y, Alignment Counter, Landing
        Status (DESCENDING / PAUSED-REALIGNING / HOVER / TOUCHDOWN), and
        Current Commanded Velocity. Read-only.
        """
        landing_states = (
            STATE_PRECISION_ALIGN,
            STATE_CONTROLLED_DESCENT,
            STATE_FINAL_LANDING,
            STATE_MISSION_COMPLETE,
        )
        if self.state not in landing_states:
            return

        if self.state == STATE_CONTROLLED_DESCENT:
            if self.descent_paused:
                landing_status, vz_cmd = 'PAUSED-REALIGNING', 0.0
            elif self.descent_hover_ready:
                landing_status, vz_cmd = 'HOVER', 0.0
            else:
                landing_status, vz_cmd = 'DESCENDING', DESCENT_SPEED
        elif self.state == STATE_FINAL_LANDING:
            if self.touchdown_occurred:
                landing_status, vz_cmd = 'TOUCHDOWN', 0.0
            elif self.descent_paused:
                landing_status, vz_cmd = 'PAUSED-REALIGNING', 0.0
            elif self.altitude > 0.25:
                landing_status, vz_cmd = 'DESCENDING', -0.08
            elif self.altitude > FINAL_LANDING_ALTITUDE:
                landing_status, vz_cmd = 'DESCENDING', -0.05
            else:
                landing_status, vz_cmd = 'DESCENDING', -0.02
        elif self.state == STATE_MISSION_COMPLETE:
            landing_status, vz_cmd = 'MISSION COMPLETE', 0.0
        else:
            landing_status, vz_cmd = 'ALIGNING', 0.0

        self.get_logger().info(
            f'[LAND DBG] State={self.state} '
            f'Alt={self.altitude:.3f}m '
            f'VertVel={self._vertical_velocity:.3f}m/s '
            f'PixelErrX={self.landing_pixel_err_x}px '
            f'PixelErrY={self.landing_pixel_err_y}px '
            f'LinearX={self._align_vx:.3f}m/s '
            f'LinearY={self._align_vy:.3f}m/s '
            f'CmdVertZ={vz_cmd:.2f}m/s '
            f'AlignCnt={self.alignment_counter}/{ALIGN_LOCK_FRAMES} '
            f'LandingStatus={landing_status} '
            f'Pad={self.landing_pad_found} '
            f'wp={self.current_waypoint}'
        )

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

        # ── V2.2: Precision Landing — detection-only, runs after existing
        # goal pipeline, never modifies goal_visible/goal_cx/goal_area/etc.
        self._detect_landing_pad(frame, hsv)
        self._draw_precision_landing_debug(frame)

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
    # V2.2 — Precision Landing pad detection (DOWN CAMERA, DETECTION ONLY)
    # ─────────────────────────────────────────────────────────────────────────
    # This pipeline is fully independent of _detect_goal() / goal_* state.
    # It does not move the drone, does not change self.state, and is not
    # consulted anywhere in control_loop() or _handle_v2_states(). It exists
    # purely to detect and visualize the red landing marker for this version.

    def _detect_landing_pad(self, frame, hsv):
        """
        Detect the red landing pad marker using HSV thresholding on the
        down camera frame and compute its contour area, centroid, and
        minimum enclosing circle, plus pixel error relative to image centre.

        Populates (read-only outside this function)
        ---------------------------------------------
        self.landing_pad_found   : bool
        self.landing_pad_cx/cy   : int   — centroid pixel coordinates
        self.landing_pad_radius  : int   — enclosing circle radius (px)
        self.landing_pad_area    : float — contour area (px^2)
        self.landing_pixel_err_x : int   — cx - image_center_x
        self.landing_pixel_err_y : int   — cy - image_center_y

        Does NOT touch goal_visible, goal_cx, goal_area, or any other
        navigation variable. Does NOT publish cmd_vel. Detection only.
        """
        mask1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
        mask2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN,
                                    np.ones((5, 5), np.uint8))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE,
                                    np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(
            red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        img_cx = self.down_img_width  // 2
        img_cy = self.down_img_height // 2

        self.landing_pad_found   = False
        self.landing_pad_cx      = img_cx
        self.landing_pad_cy      = img_cy
        self.landing_pad_radius  = 0
        self.landing_pad_area    = 0.0
        self.landing_pixel_err_x = 0
        self.landing_pixel_err_y = 0
        self._last_landing_contour = None
        self._last_landing_circle  = None  # ((cx, cy), radius) for drawing

        if not contours:
            return

        # Largest red contour = landing pad marker
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < GOAL_MIN_AREA:
            return

        M = cv2.moments(largest)
        if M['m00'] == 0:
            return

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        (circ_x, circ_y), radius = cv2.minEnclosingCircle(largest)

        self.landing_pad_found   = True
        self.landing_pad_cx      = cx
        self.landing_pad_cy      = cy
        self.landing_pad_radius  = int(radius)
        self.landing_pad_area    = area
        self.landing_pixel_err_x = cx - img_cx
        self.landing_pixel_err_y = cy - img_cy
        self._last_landing_contour = largest
        self._last_landing_circle  = ((int(circ_x), int(circ_y)), int(radius))

    def _draw_precision_landing_debug(self, frame):
        """
        Step 6 — Render the "Precision Landing" OpenCV window.

        V2.5: adds Current X/Y Velocity, Controller Mode, and P-controller
        output (Problem 7), and shows "INITIALIZING..." instead of a
        misleading 0.000 m altitude before the first odom message (Problem 4).

        Read-only — does not alter any navigation state or publish commands.
        """
        debug = frame.copy()
        h, w = debug.shape[:2]
        img_cx, img_cy = w // 2, h // 2

        # Image-centre crosshair for visual reference
        cv2.drawMarker(debug, (img_cx, img_cy), (255, 255, 255),
                        markerType=cv2.MARKER_CROSS, markerSize=20, thickness=1)

        if self.landing_pad_found:
            if self._last_landing_contour is not None:
                cv2.drawContours(debug, [self._last_landing_contour], -1,
                                  (0, 255, 0), 2)
            if self._last_landing_circle is not None:
                (circ_cx, circ_cy), circ_r = self._last_landing_circle
                cv2.circle(debug, (circ_cx, circ_cy), circ_r, (255, 0, 0), 2)
            cv2.circle(debug, (self.landing_pad_cx, self.landing_pad_cy),
                       5, (0, 0, 255), -1)
            pad_str, pad_col = 'FOUND', (0, 255, 0)
        else:
            pad_str, pad_col = 'NOT FOUND', (0, 0, 255)

        # V2.5 — Problem 4: self.altitude itself is untouched; only the
        # displayed string changes until the first /drone/odom msg arrives.
        alt_str = f'{self.altitude:.3f} m' if self._odom_received else 'INITIALIZING...'

        # ── Fixed information rows (always shown) ─────────────────────────────
        cv2.putText(debug, f'Landing Pad : {pad_str}',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, pad_col, 2)
        cv2.putText(debug, f'Mission State : {self.state}',
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
        cv2.putText(debug, f'Altitude : {alt_str}',
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
        cv2.putText(debug, f'Pixel Error X : {self.landing_pixel_err_x}',
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 0), 1)
        cv2.putText(debug, f'Pixel Error Y : {self.landing_pixel_err_y}',
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 0), 1)
        cv2.putText(debug, f'Alignment Counter : {self.alignment_counter} / 10',
                    (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 0), 1)

        # ── V2.5 — Problem 7: P-controller telemetry ─────────────────────────
        cv2.putText(debug, f'Controller Mode : P-CONTROLLER (Kp={PRECISION_KP})',
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 220), 1)
        cv2.putText(debug, f'P Output (x,y) : {self._precision_p_out_y:.3f}, {self._precision_p_out_x:.3f} m/s',
                    (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 220), 1)
        cv2.putText(debug, f'Current X Velocity : {self._align_vx:.3f} m/s',
                    (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
        cv2.putText(debug, f'Current Y Velocity : {self._align_vy:.3f} m/s',
                    (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

        # ── State-specific rows ────────────────────────────────────────────────
        vz_display   = 0.0
        phase_str    = self.state
        status_str   = 'ALIGNING'
        status_col   = (0, 165, 255)

        if self.state == STATE_PRECISION_ALIGN:
            x_ok = abs(self.landing_pixel_err_x) <= PRECISION_PIXEL_TOL
            y_ok = abs(self.landing_pixel_err_y) <= PRECISION_PIXEL_TOL
            phase_str  = 'ALIGNING'
            status_str = 'CENTERED' if (self.landing_pad_found and x_ok and y_ok) else 'ALIGNING'
            status_col = (0, 255, 0) if status_str == 'CENTERED' else (0, 165, 255)

        elif self.state == STATE_CONTROLLED_DESCENT:
            phase_str = 'CONTROLLED DESCENT'
            if self.descent_paused:
                vz_display, status_str, status_col = 0.0, 'PAUSED-REALIGNING', (0, 0, 255)
            elif self.descent_hover_ready:
                vz_display, status_str, status_col = 0.0, 'HOVER', (0, 165, 255)
            else:
                vz_display, status_str, status_col = DESCENT_SPEED, 'DESCENDING', (0, 165, 255)

        elif self.state == STATE_FINAL_LANDING:
            if self.touchdown_occurred:
                vz_display = 0.0
                phase_str  = 'TOUCHDOWN'
                status_str = 'TOUCHDOWN'
                status_col = (0, 255, 255)
            elif self.descent_paused:
                vz_display = 0.0
                phase_str  = 'FINAL LANDING'
                status_str = 'PAUSED-REALIGNING'
                status_col = (0, 0, 255)
            else:
                if self.altitude > 0.25:
                    vz_display = -0.08
                elif self.altitude > FINAL_LANDING_ALTITUDE:
                    vz_display = -0.05
                else:
                    vz_display = -0.02
                phase_str  = 'FINAL LANDING'
                status_str = 'LANDING'
                status_col = (0, 165, 255)

        elif self.state == STATE_MISSION_COMPLETE:
            phase_str  = 'MISSION COMPLETE'
            status_str = 'MISSION COMPLETE'
            status_col = (0, 255, 0)

        cv2.putText(debug, f'Vertical Speed : {vz_display:.2f} m/s',
                    (10, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
        cv2.putText(debug, f'Landing Phase : {phase_str}',
                    (10, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 200, 255), 1)
        cv2.putText(debug, f'Landing Status : {status_str}',
                    (10, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_col, 2)

        cv2.imshow('Precision Landing', debug)
        cv2.waitKey(1)

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
    # V2.2 Part 2 — Precision alignment using landing-pad pixel error
    # (DOWN CAMERA, landing detection only — does not modify
    # _detect_landing_pad() or any V1 goal-alignment logic)
    # ─────────────────────────────────────────────────────────────────────────

    def precision_align(self) -> tuple:
        """
        V2.7 — Proportional (P) controller for landing-pad pixel-error
        centring, using ONE FIXED coordinate mapping. The automatic axis-flip
        system (AXIS_FLIP_CONSECUTIVE_FRAMES, growth counters, per-axis
        flip flags) has been removed entirely per the V2.4 Final Stable spec.

        P controller
        ------------
        velocity = PRECISION_KP * pixel_error, clamped to ±PRECISION_MAX_SPEED,
        with a ±PRECISION_DEADBAND_PX deadband (commanded 0.0 inside it).

        Fixed axis mapping
        -------------------
        Image X → body Y (left/right):
            error_x > 0 (pad right of centre) → move right  → linear.y negative
        Image Y → body X (forward/back):
            error_y > 0 (pad below centre)    → move backward → linear.x negative

        V2.4 Final Stable fix — this axis previously used a *positive*
        sign (linear.x = +p_out_y) while the X-axis used a negative sign
        (linear.y = -p_out_x). Both pixel errors come from the same
        optical-frame transform into the same body frame, so they must
        share the same sign convention. The un-negated Y term meant the
        P-controller commanded motion in the direction that *increased*
        pixel_err_y, which — combined with saturation at
        PRECISION_MAX_SPEED — produced the observed monotonic drift
        (pixel_err_y growing while cmd.linear.x stayed pinned at +0.120)
        instead of convergence. Flipping the sign to match the X-axis
        convention resolves it.

        Pad-lost grace period
        ------------------------------------
        If landing_pad_found goes False, the last good correction
        (self._last_precision_cmd) is reused for up to PAD_LOST_GRACE_TICKS
        (0.5 s) before the drone is commanded to stop. This absorbs brief
        detection dropouts without interrupting the descent/alignment.

        Returns
        -------
        (cmd: Twist, aligned: bool)
        aligned is True only when both axes are within PRECISION_PIXEL_TOL
        — the single tolerance used everywhere in the landing controller.
        """
        # ── pad-lost grace period ────────────────────────────────────────────
        if self.landing_pad_found:
            self._align_pad_lost_ticks = 0
        else:
            self._align_pad_lost_ticks += 1
            if self._align_pad_lost_ticks <= PAD_LOST_GRACE_TICKS:
                # Still inside the 0.5 s grace window — reuse the last good
                # correction rather than stopping immediately.
                return self._last_precision_cmd, False
            # Grace period exhausted — pad has been missing too long, stop.
            self._align_vx = 0.0
            self._align_vy = 0.0
            self._precision_p_out_x = 0.0
            self._precision_p_out_y = 0.0
            self._last_precision_cmd = Twist()
            return Twist(), False

        cmd = Twist()

        err_x = self.landing_pixel_err_x
        err_y = self.landing_pixel_err_y

        x_aligned = abs(err_x) <= PRECISION_PIXEL_TOL
        y_aligned = abs(err_y) <= PRECISION_PIXEL_TOL

        # ── X axis P term (image X → body Y), fixed mapping ─────────────────────
        if abs(err_x) <= PRECISION_DEADBAND_PX:
            p_out_x = 0.0
        else:
            p_out_x = float(np.clip(PRECISION_KP * err_x,
                                     -PRECISION_MAX_SPEED, PRECISION_MAX_SPEED))
        cmd.linear.y = -p_out_x

        # ── Y axis P term (image Y → body X), fixed mapping ─────────────────────
        if abs(err_y) <= PRECISION_DEADBAND_PX:
            p_out_y = 0.0
        else:
            p_out_y = float(np.clip(PRECISION_KP * err_y,
                                     -PRECISION_MAX_SPEED, PRECISION_MAX_SPEED))
        cmd.linear.x = -p_out_y

        cmd.linear.z  = 0.0   # no descent — alignment only
        cmd.angular.z = 0.0   # no rotation

        # Telemetry for the Precision Landing debug window / 1 Hz log
        self._precision_p_out_x = p_out_x
        self._precision_p_out_y = p_out_y
        self._align_vx = cmd.linear.x
        self._align_vy = cmd.linear.y
        self._last_precision_cmd = cmd

        aligned = x_aligned and y_aligned
        return cmd, aligned

    # ─────────────────────────────────────────────────────────────────────────
    # V2.7 Part 1 — Incremental (stepped) descent primitive, shared by both
    # controlled_descent() and final_landing(). Replaces continuous descent
    # entirely: descend DESCENT_STEP_METERS (~5 cm), stop, re-check
    # alignment, then either take the next step or pause and re-align.
    # Uses ONLY PRECISION_PIXEL_TOL for alignment gating (single tolerance
    # everywhere — DESCENT_ALIGN_TOL has been removed).
    # ─────────────────────────────────────────────────────────────────────────

    def _stepped_descent_command(self, step_speed: float) -> tuple:
        """
        V2.7 — Shared incremental-descent primitive.

        Sequence per the spec
        ----------------------
        Align → descend ~5 cm → stop → check alignment →
            if centered: descend another 5 cm
            else: correct X/Y (pause vertical motion) → repeat

        Alignment gating uses PRECISION_PIXEL_TOL exclusively (the same
        tolerance used by precision_align()/STATE_PRECISION_ALIGN — no
        looser tolerance is ever used during descent). If either pixel
        error exceeds PRECISION_PIXEL_TOL, or the pad disappears, vertical
        motion pauses immediately ('ALIGNMENT LOST', logged once) and only
        resumes once ALIGN_STABLE_FRAMES consecutive aligned frames are
        seen again ('ALIGNMENT RECOVERED', logged once) — this is what
        stops the drone from landing beside the pad.

        Returns
        -------
        (cmd: Twist, stepping: bool)
        stepping is True while a step is actively in progress (still moving
        down toward this step's target altitude).
        """
        cmd, _ = self.precision_align()

        misaligned = (abs(self.landing_pixel_err_x) > PRECISION_PIXEL_TOL or
                      abs(self.landing_pixel_err_y) > PRECISION_PIXEL_TOL or
                      not self.landing_pad_found)

        if misaligned:
            if not self.descent_paused:
                self.descent_paused = True
                self.get_logger().warn('ALIGNMENT LOST')
            self.descent_realign_lock = 0
            self.descent_step_active  = False   # abandon the in-progress step
            cmd.linear.z  = 0.0
            cmd.angular.z = 0.0
            return cmd, False

        if self.descent_paused:
            self.descent_realign_lock += 1
            if self.descent_realign_lock >= ALIGN_STABLE_FRAMES:
                self.descent_paused = False
                self.get_logger().info('ALIGNMENT RECOVERED')
            else:
                cmd.linear.z  = 0.0
                cmd.angular.z = 0.0
                return cmd, False

        # ── Aligned and stable — take (or continue) one fixed-size step ────────
        if not self.descent_step_active:
            self.descent_step_active = True
            self.descent_step_target_altitude = self.altitude - DESCENT_STEP_METERS

        if self.altitude <= self.descent_step_target_altitude:
            # Step complete — stop here; the NEXT tick re-checks alignment
            # before a new 5 cm step is allowed to start.
            self.descent_step_active = False
            cmd.linear.z = 0.0
        else:
            cmd.linear.z = step_speed

        cmd.angular.z = 0.0
        return cmd, self.descent_step_active

    # ─────────────────────────────────────────────────────────────────────────
    # V2.3 Part 1 — Controlled descent (coarse incremental descent stage)
    # ─────────────────────────────────────────────────────────────────────────

    def controlled_descent(self) -> tuple:
        """
        V2.7 — Coarse descent stage: repeatedly takes DESCENT_STEP_METERS
        (~5 cm) steps via _stepped_descent_command(), pausing to re-align
        between every step. Hands off to STATE_FINAL_LANDING once altitude
        reaches DESCENT_ALTITUDE.

        Returns
        -------
        (cmd: Twist, hover_ready: bool)
        """
        cmd, _ = self._stepped_descent_command(DESCENT_SPEED)

        hover_ready = self.altitude <= DESCENT_ALTITUDE
        if hover_ready:
            cmd.linear.z = 0.0
            self.descent_step_active = False   # clean handoff to FINAL_LANDING

        return cmd, hover_ready

    # ─────────────────────────────────────────────────────────────────────────
    # V2.3 Part 2 — Final soft landing (fine incremental descent to touchdown,
    # reusing the same stepping primitive as controlled_descent())
    # ─────────────────────────────────────────────────────────────────────────

    def final_landing(self) -> tuple:
        """
        V2.7 — Final incremental descent to touchdown: same 5 cm stepping
        primitive as controlled_descent(), using the slower
        FINAL_DESCENT_STEP_SPEED. Touchdown requires ALL of:
        altitude below TOUCHDOWN_ALTITUDE, vertical velocity ~0, the
        landing pad still detected, both pixel errors inside
        PRECISION_PIXEL_TOL, and stable horizontal position — sustained for
        TOUCHDOWN_CONFIRM_TICKS consecutive ticks before touchdown is
        confirmed.

        Returns
        -------
        (cmd: Twist, touchdown_confirmed: bool)
        """
        cmd, _ = self._stepped_descent_command(FINAL_DESCENT_STEP_SPEED)

        altitude_ok = self.altitude <= TOUCHDOWN_ALTITUDE
        vz_ok       = abs(self._vertical_velocity) <= TOUCHDOWN_VZ_TOL
        pad_ok      = self.landing_pad_found
        x_ok        = abs(self.landing_pixel_err_x) <= PRECISION_PIXEL_TOL
        y_ok        = abs(self.landing_pixel_err_y) <= PRECISION_PIXEL_TOL
        drift_ok    = self._horizontal_drift <= TOUCHDOWN_XY_TOL

        if altitude_ok and vz_ok and pad_ok and x_ok and y_ok and drift_ok:
            self.touchdown_confirm_ticks += 1
        else:
            self.touchdown_confirm_ticks = 0

        touchdown_confirmed = self.touchdown_confirm_ticks >= TOUCHDOWN_CONFIRM_TICKS

        if touchdown_confirmed:
            cmd = Twist()   # touchdown confirmed — zero every axis immediately

        cmd.angular.z = 0.0
        return cmd, touchdown_confirmed

    # ─────────────────────────────────────────────────────────────────────────
    # Main control loop (unchanged logic)
    # ─────────────────────────────────────────────────────────────────────────

    def control_loop(self):
        cmd = Twist()
        self._update_landing_telemetry()   # V2.6 — cheap, state-independent

        # Publish state every loop tick
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

        # vz is always 0.0 after takeoff — no altitude correction
        vz = 0.0  # noqa: F841

        # Timer-based takeoff — 60 ticks at 20 Hz = 3 seconds
        # V2: after takeoff, enter PLAN_PATH instead of FIND_GOAL
        if self.state == STATE_TAKEOFF:
            # ── Reset all landing state on every takeoff for a clean start ─────
            self.alignment_counter   = 0
            self.touchdown_occurred  = False
            self.landing_timer_ticks = 0
            self.descent_active      = False
            self.descent_hover_ready = False

            # V2.6 — reset alignment-gated descent + touchdown state
            self.descent_paused          = False
            self.descent_realign_lock    = 0
            self.touchdown_confirm_ticks = 0

            # V2.7 — reset incremental (stepped) descent state
            self.descent_step_active          = False
            self.descent_step_target_altitude = None

            # V2.5b — reset pad-lost grace state
            # (auto axis-flip system removed — nothing to reset there)
            self._align_pad_lost_ticks = 0
            self._last_precision_cmd   = Twist()

            cmd.linear.z = VERT_SPEED
            self.takeoff_ticks += 1
            if self.takeoff_ticks >= 60:
                self.state = STATE_PLAN_PATH
                self.get_logger().info('Takeoff complete → Planning Global Path...')
                enable_msg = Bool()
                enable_msg.data = True
                self.enable_pub.publish(enable_msg)
            self.cmd_pub.publish(cmd)
            return

        if self.state == STATE_MISSION_DONE:
            cmd.linear.z = 0.0
            self.cmd_pub.publish(cmd)
            return

        # ── V2.3 Part 2: terminal state — never re-enters navigation ───────────
        if self.state == STATE_MISSION_COMPLETE:
            if not self.mission_complete_logged:
                self.get_logger().info('===================================')
                self.get_logger().info('MISSION SUCCESSFUL')
                self.get_logger().info('PRECISION LANDING COMPLETE')
                self.get_logger().info('===================================')
                self.mission_complete_logged = True
            self.cmd_pub.publish(Twist())
            return

        # ── V2 path-planning state hook ───────────────────────────────────────
        # Runs before goal-alignment and obstacle logic so PLAN_PATH /
        # FOLLOW_PATH / RETURN_TO_PATH are fully handled here.
        # Returns (True, cmd) when the current state is a V2 path state;
        # returns (False, zero) when the current state belongs to V1.
        v2_handled, v2_cmd = self._handle_v2_states()
        if v2_handled:
            self.cmd_pub.publish(v2_cmd)
            return

        # ── Enter alignment phase only when goal is close enough ──────────────
        # Guard: never interrupt a landing or path state with alignment.
        if (self.goal_visible
                and self.goal_area > ALIGN_TRIGGER_AREA
                and self.state not in (STATE_PLAN_PATH,
                                       STATE_FOLLOW_PATH,
                                       STATE_RETURN_TO_PATH,
                                       STATE_MISSION_DONE,
                                       STATE_PRECISION_ALIGN,
                                       STATE_CONTROLLED_DESCENT,
                                       STATE_FINAL_LANDING)):
            self.state = STATE_ALIGN_GOAL
            self.get_logger().info('Goal Alignment')

        # ── V2.2 Part 2: enter precision alignment once goal area crosses the
        # mission-completion threshold — takes over from STATE_ALIGN_GOAL.
        if (self.goal_visible
                and self.goal_area > GOAL_AREA_THRESHOLD
                and self.state == STATE_ALIGN_GOAL):
            self.state = STATE_PRECISION_ALIGN
            # V2.5b — clear the pad-lost grace counter for this new attempt.
            self._align_pad_lost_ticks = 0
            self.get_logger().info('Precision Alignment')

        # ── STATE_ALIGN_GOAL — fine-position over goal before completing ──────
        if self.state == STATE_ALIGN_GOAL:

            # Goal lost during alignment → resume searching
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
                    self.get_logger().info('Mission Complete')
                    status.data = f'{STATE_MISSION_DONE}|MISSION_COMPLETE'
                    self.status_pub.publish(status)
                    self.cmd_pub.publish(Twist())   # zero velocity — hover in place
                    return

                self.cmd_pub.publish(cmd)
                return

        # ── V2.5b: STATE_PRECISION_ALIGN — pixel-error centring only ────────────
        # Uses landing_pixel_err_x/y (populated by _detect_landing_pad in the
        # down camera callback). Does NOT descend, does NOT land. Transitions
        # to STATE_CONTROLLED_DESCENT once ALIGN_LOCK_FRAMES (10) consecutive
        # aligned frames are seen. Never times out — alignment continues
        # indefinitely rather than forcing an unaligned descent.
        if self.state == STATE_PRECISION_ALIGN:

            # precision_align() itself absorbs brief pad-loss (0.5 s grace
            # period), so it is safe to call unconditionally here.
            cmd, aligned = self.precision_align()

            # Pad missing beyond the grace period → give up and search again.
            if not self.landing_pad_found and self._align_pad_lost_ticks > PAD_LOST_GRACE_TICKS:
                self.alignment_counter = 0
                self.state = STATE_FIND_GOAL
                self.cmd_pub.publish(Twist())
                return

            # ── Counter-based lock (10 consecutive aligned frames) ─────────────
            if aligned:
                self.alignment_counter += 1
            else:
                self.alignment_counter = 0

            # ── Hand off to controlled descent once locked in ──────────────────
            if self.alignment_counter >= ALIGN_LOCK_FRAMES:
                self.get_logger().info('===================================')
                self.get_logger().info('PRECISION ALIGNMENT COMPLETE')
                self.get_logger().info('===================================')
                self.state = STATE_CONTROLLED_DESCENT
                self.get_logger().info('===================================')
                self.get_logger().info('CONTROLLED DESCENT')
                self.get_logger().info('===================================')
                self.cmd_pub.publish(Twist())   # one zero-velocity tick at handoff
                return

            self.cmd_pub.publish(cmd)
            return

        # ── V2.3 Part 1: STATE_CONTROLLED_DESCENT — descend while continuously
        # re-correcting X/Y via precision_align(). Stops descending (hovers)
        # once self.altitude <= DESCENT_ALTITUDE. Does NOT land,
        # does NOT stop the motors, does NOT finish the mission.
        if self.state == STATE_CONTROLLED_DESCENT:

            # Landing pad lost during descent → immediately stop descent and
            # hover; resume once the pad is detected again (handled by the
            # cmd, hover_ready = controlled_descent() guard returning zero
            # Twist while landing_pad_found is False).
            cmd, hover_ready = self.controlled_descent()
            self.descent_active      = not hover_ready and self.landing_pad_found
            self.descent_hover_ready = hover_ready

            # ── V2.3 Part 2: hand off to final landing once hover_ready ────────
            if hover_ready:
                self.state = STATE_FINAL_LANDING
                self.get_logger().info('===================================')
                self.get_logger().info('FINAL LANDING')
                self.get_logger().info('===================================')
                self.cmd_pub.publish(Twist())   # one zero-velocity tick at handoff
                return

            self.cmd_pub.publish(cmd)
            return

        # ── V2.3 Part 2: STATE_FINAL_LANDING — incremental descent + continuous
        # X/Y correction via final_landing() (reuses precision_align() and
        # _stepped_descent_command()). Declares TOUCHDOWN once the
        # multi-condition check in final_landing() confirms it, then keeps
        # checking stability post-touchdown and only disables the motors
        # (and prints MISSION SUCCESSFUL) once stable for ~1 s straight,
        # before handing off to STATE_MISSION_COMPLETE.
        if self.state == STATE_FINAL_LANDING:

            cmd, touchdown = self.final_landing()

            # ── One-shot touchdown detection ────────────────────────────────────
            # touchdown_occurred latches True the first time final_landing()
            # confirms touchdown and never resets within this mission.
            if touchdown and not self.touchdown_occurred:
                self.touchdown_occurred = True
                self.get_logger().info('============================')
                self.get_logger().info('TOUCHDOWN')
                self.get_logger().info('============================')

            # ── Post-touchdown: hold zero velocity, KEEP checking stability, and
            # only disable motors / declare the mission successful once stable
            # for ~1 s straight. Never disable motors immediately on touchdown.
            if self.touchdown_occurred:
                self.cmd_pub.publish(Twist())   # zero velocity every tick

                still_stable = (abs(self._vertical_velocity) <= TOUCHDOWN_VZ_TOL and
                                 self._horizontal_drift <= TOUCHDOWN_XY_TOL)
                if still_stable:
                    self.landing_timer_ticks += 1
                else:
                    self.landing_timer_ticks = 0   # destabilized — keep waiting

                if self.landing_timer_ticks >= MOTOR_DISABLE_HOLD_TICKS:
                    if not self.mission_complete_logged:
                        enable_msg = Bool()
                        enable_msg.data = False
                        self.enable_pub.publish(enable_msg)
                        self.get_logger().info('============================')
                        self.get_logger().info('MISSION SUCCESSFUL')
                        self.get_logger().info('PRECISION LANDING COMPLETE')
                        self.get_logger().info('============================')
                        self.mission_complete_logged = True
                    self.state = STATE_MISSION_COMPLETE
                return

            self.cmd_pub.publish(cmd)
            return

        # ── V1 reactive obstacle avoidance (FIND_GOAL / ROTATE states) ────────
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

    # ─────────────────────────────────────────────────────────────────────────
    # V2 OCCUPANCY GRID — Phase 1 infrastructure (passive, read-only)
    # ─────────────────────────────────────────────────────────────────────────

    def build_occupancy_grid(self) -> list:
        """
        Build and return a 2-D occupancy grid from self.static_obstacles.

        Grid layout
        -----------
        self.occupancy_grid[row][col]
          row  : world Y axis  (row 0 = grid_origin_y,  increases northward)
          col  : world X axis  (col 0 = grid_origin_x,  increases eastward)

        Cell values
        -----------
          0 = free
          1 = occupied (static obstacle or inflation shell)

        Inflation
        ---------
        Every obstacle cell is expanded by self.grid_inflation cells in all
        8 directions to create a safety margin around each chair.

        Returns
        -------
        list[list[int]]  shape (grid_rows × grid_cols), all values 0 or 1.
        """
        rows = self.grid_rows
        cols = self.grid_cols
        pad  = self.grid_inflation

        # Initialise every cell as free
        grid = [[0] * cols for _ in range(rows)]

        for (wx, wy) in self.static_obstacles:
            # Convert world position to grid indices
            col, row = self.world_to_grid(wx, wy)

            # Skip obstacles that fall outside the grid boundaries
            if col is None or row is None:
                self.get_logger().warn(
                    f'[OccGrid] Static obstacle ({wx}, {wy}) is outside grid bounds — skipped.')
                continue

            # Mark the obstacle cell and its inflation neighbourhood
            for dr in range(-pad, pad + 1):
                for dc in range(-pad, pad + 1):
                    r = row + dr
                    c = col + dc
                    if 0 <= r < rows and 0 <= c < cols:
                        grid[r][c] = 1

        return grid

    def world_to_grid(self, wx: float, wy: float):
        """
        Convert a world-frame (x, y) position to (col, row) grid indices.

        Parameters
        ----------
        wx : float  — world X coordinate (metres)
        wy : float  — world Y coordinate (metres)

        Returns
        -------
        (col, row) : tuple[int, int]  — zero-based grid indices, or
        (None, None)                  — if the position is outside the grid.
        """
        col = int((wx - self.grid_origin_x) / self.grid_cell_size)
        row = int((wy - self.grid_origin_y) / self.grid_cell_size)

        if 0 <= col < self.grid_cols and 0 <= row < self.grid_rows:
            return col, row
        return None, None

    def grid_to_world(self, col: int, row: int):
        """
        Convert (col, row) grid indices to the world-frame centre of that cell.

        Parameters
        ----------
        col : int — grid column index (X direction)
        row : int — grid row index    (Y direction)

        Returns
        -------
        (wx, wy) : tuple[float, float]  — world coordinates of the cell centre,
                   or (None, None) if the indices are out of bounds.
        """
        if not (0 <= col < self.grid_cols and 0 <= row < self.grid_rows):
            return None, None

        wx = self.grid_origin_x + (col + 0.5) * self.grid_cell_size
        wy = self.grid_origin_y + (row + 0.5) * self.grid_cell_size
        return wx, wy

    def print_grid_debug(self):
        """
        Print a concise occupancy grid summary to the ROS 2 terminal.

        Outputs
        -------
        - One-line header with grid dimensions and obstacle count
        - Count of occupied cells (after inflation)
        - ASCII map: '#' = occupied, '.' = free
          Rows are printed top-to-bottom (highest Y first) for readability.

        This is the only visualisation for Phase 1.
        No OpenCV drawing and no RViz markers are used.
        """
        occupied_cells = sum(
            self.occupancy_grid[r][c]
            for r in range(self.grid_rows)
            for c in range(self.grid_cols)
        )

        self.get_logger().info('=' * 50)
        self.get_logger().info('Occupancy Grid Built')
        self.get_logger().info(f'  Size     : {self.grid_cols} x {self.grid_rows}  |  Cell : {self.grid_cell_size} m')
        self.get_logger().info(f'  Obstacles: {len(self.static_obstacles)} static  |  Occupied cells: {occupied_cells}  |  Inflation: {self.grid_inflation}')
        self.get_logger().info('-' * 50)

        # Print ASCII map — rows drawn top-to-bottom (high Y → low Y)
        for r in range(self.grid_rows - 1, -1, -1):
            row_str = ''.join(
                '#' if self.occupancy_grid[r][c] == 1 else '.'
                for c in range(self.grid_cols)
            )
            self.get_logger().info(f'  Y{r:02d} |{row_str}|')

        self.get_logger().info('       ' + ''.join(
            str(c % 10) for c in range(self.grid_cols)))
        self.get_logger().info('         X -->')
        self.get_logger().info('=' * 50)

    # ─────────────────────────────────────────────────────────────────────────
    # V2.1 — A* Planner Visualization (display-only, no logic impact)
    # ─────────────────────────────────────────────────────────────────────────

    def draw_astar_visualization(self):
        """
        Render a real-time OpenCV window showing the occupancy grid, the cells
        explored by the last A* run, the final path, the drone's live position,
        and the fixed goal cell — plus a text overlay of mission/path stats.

        This function is purely visual. It reads existing data already
        produced elsewhere (self.occupancy_grid, self._astar_explored_cells,
        self._astar_grid_path, self.path, self.current_x/y, self.state,
        self.current_waypoint) and never writes back to navigation state,
        never re-runs A*, and never regenerates the occupancy grid.

        Colours
        -------
        White  : free cell
        Black  : occupied cell
        Blue   : cell expanded by the most recent A* search
        Yellow : final A* path (drawn as connected lines)
        Green  : current drone grid position
        Red    : fixed goal grid position

        Runs at ~10 FPS via a dedicated timer, independent of the 20 Hz
        control loop, so it never slows down navigation.
        """
        cell_px = self._viz_cell_px
        cols    = self.grid_cols
        rows    = self.grid_rows

        img_w = cols * cell_px
        img_h = rows * cell_px + 130   # extra space at bottom for text overlay

        canvas = np.full((img_h, img_w, 3), 255, dtype=np.uint8)  # white background

        # ── Draw occupancy grid: free = white (already filled), obstacle = black ─
        for r in range(rows):
            for c in range(cols):
                if self.occupancy_grid[r][c] == 1:
                    # Flip row so higher world Y draws toward the top of the window
                    draw_r = rows - 1 - r
                    y0 = draw_r * cell_px
                    x0 = c * cell_px
                    cv2.rectangle(canvas, (x0, y0),
                                  (x0 + cell_px, y0 + cell_px),
                                  (0, 0, 0), -1)

        # ── Draw cells explored by the most recent A* search (blue) ───────────
        for (c, r) in self._astar_explored_cells:
            draw_r = rows - 1 - r
            y0 = draw_r * cell_px
            x0 = c * cell_px
            # Only tint free cells so obstacles stay visibly black
            if self.occupancy_grid[r][c] == 0:
                cv2.rectangle(canvas, (x0, y0),
                              (x0 + cell_px, y0 + cell_px),
                              (255, 150, 50), -1)   # blue (BGR)

        # ── Draw grid lines for readability ────────────────────────────────────
        for c in range(cols + 1):
            x = c * cell_px
            cv2.line(canvas, (x, 0), (x, rows * cell_px), (200, 200, 200), 1)
        for r in range(rows + 1):
            y = r * cell_px
            cv2.line(canvas, (0, y), (img_w, y), (200, 200, 200), 1)

        # ── Draw final A* path as connected yellow lines ───────────────────────
        if len(self._astar_grid_path) > 1:
            pts = []
            for (c, r) in self._astar_grid_path:
                draw_r = rows - 1 - r
                px = c * cell_px + cell_px // 2
                py = draw_r * cell_px + cell_px // 2
                pts.append((px, py))
            for i in range(len(pts) - 1):
                cv2.line(canvas, pts[i], pts[i + 1], (0, 255, 255), 2)  # yellow (BGR)

        # ── Draw goal position (red) ────────────────────────────────────────────
        goal_col, goal_row = self.world_to_grid(
            DroneNavigator.GOAL_WORLD_X, DroneNavigator.GOAL_WORLD_Y)
        if goal_col is not None:
            draw_r = rows - 1 - goal_row
            gx = goal_col * cell_px + cell_px // 2
            gy = draw_r * cell_px + cell_px // 2
            cv2.circle(canvas, (gx, gy), cell_px // 2 - 2, (0, 0, 255), -1)  # red

        # ── Draw drone's live position (green) ──────────────────────────────────
        drone_col, drone_row = self.world_to_grid(self.current_x, self.current_y)
        if drone_col is not None:
            draw_r = rows - 1 - drone_row
            dx = drone_col * cell_px + cell_px // 2
            dy = draw_r * cell_px + cell_px // 2
            cv2.circle(canvas, (dx, dy), cell_px // 2 - 2, (0, 255, 0), -1)  # green

        # ── Text overlay: mission / path stats ──────────────────────────────────
        overlay_y0 = rows * cell_px + 5
        total_wp   = len(self.path) if self.path else 0
        cur_wp     = min(self.current_waypoint, total_wp) if total_wp else 0
        path_len   = len(self._astar_grid_path)

        lines = [
            f'Mission : {self.state}',
            f'Waypoint : {cur_wp} / {total_wp}',
            f'Path Length : {path_len}',
            f'Expanded Nodes : {self._astar_nodes_expanded}',
            f'Planning Time : {self._astar_planning_ms:.0f} ms',
        ]
        for i, text in enumerate(lines):
            cv2.putText(canvas, text, (8, overlay_y0 + 18 * i + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow('A* Planner', canvas)
        cv2.waitKey(1)



    # Fixed goal world position — red pad centre from drone_arena.sdf
    GOAL_WORLD_X = 9.0
    GOAL_WORLD_Y = 0.0

    def heuristic(self, col: int, row: int, goal_col: int, goal_row: int) -> float:
        """
        Octile distance heuristic — admissible for 8-direction movement
        with straight cost 1.0 and diagonal cost sqrt(2).

        Parameters
        ----------
        col, row           : current cell indices
        goal_col, goal_row : goal cell indices

        Returns
        -------
        float — lower-bound cost estimate to the goal
        """
        dx = abs(col - goal_col)
        dy = abs(row - goal_row)
        return (dx + dy) + (math.sqrt(2) - 2.0) * min(dx, dy)

    def get_neighbors(self, col: int, row: int) -> list:
        """
        Return all traversable 8-connected neighbours of cell (col, row).

        Movement costs
        --------------
        Cardinal  (N/S/E/W)   : 1.0
        Diagonal (NE/NW/SE/SW): sqrt(2) ≈ 1.414

        A neighbour is excluded when:
        - it falls outside the grid boundary, or
        - its occupancy_grid value is 1 (occupied / inflated obstacle), or
        - it is a diagonal move and either adjacent cardinal cell is occupied
          (corner-cutting prevention — the drone cannot squeeze through a
          gap that exists only at the corner of two obstacle cells).

        Corner-cutting rule (diagonal moves only)
        -----------------------------------------
        Moving from (col, row) to (col+dc, row+dr) diagonally requires
        both intermediate cardinal cells to be free:
          - (col+dc, row)    — horizontal neighbour
          - (col,    row+dr) — vertical neighbour
        If either is occupied the diagonal is skipped.

        Parameters
        ----------
        col : int — column index of the current cell
        row : int — row index of the current cell

        Returns
        -------
        list of (cost: float, neighbour_col: int, neighbour_row: int)
        """
        CARDINAL = [
            ( 0,  1, 1.0),           # N
            ( 0, -1, 1.0),           # S
            ( 1,  0, 1.0),           # E
            (-1,  0, 1.0),           # W
        ]
        DIAGONAL = [
            ( 1,  1, math.sqrt(2)),  # NE
            (-1,  1, math.sqrt(2)),  # NW
            ( 1, -1, math.sqrt(2)),  # SE
            (-1, -1, math.sqrt(2)),  # SW
        ]

        neighbours = []

        # ── Cardinal moves — no additional check needed ───────────────────────
        for dc, dr, cost in CARDINAL:
            nc, nr = col + dc, row + dr
            if 0 <= nc < self.grid_cols and 0 <= nr < self.grid_rows:
                if self.occupancy_grid[nr][nc] == 0:
                    neighbours.append((cost, nc, nr))

        # ── Diagonal moves — skip if either adjacent cardinal cell is occupied ─
        for dc, dr, cost in DIAGONAL:
            nc, nr = col + dc, row + dr
            # Target cell must be in-bounds and free
            if not (0 <= nc < self.grid_cols and 0 <= nr < self.grid_rows):
                continue
            if self.occupancy_grid[nr][nc] != 0:
                continue
            # Horizontal neighbour: same row, shifted column
            if self.occupancy_grid[row][col + dc] != 0:
                continue
            # Vertical neighbour: same column, shifted row
            if self.occupancy_grid[row + dr][col] != 0:
                continue
            neighbours.append((cost, nc, nr))

        return neighbours

    def astar(self, start_col: int, start_row: int,
              goal_col: int, goal_row: int) -> tuple:
        """
        A* search on self.occupancy_grid.

        Parameters
        ----------
        start_col, start_row : grid cell of the drone's current position
        goal_col,  goal_row  : grid cell of the fixed goal

        Returns
        -------
        (path, nodes_expanded) where path is a list of (col, row) tuples
        from start to goal inclusive, or [] if no path exists.
        """
        # Open list entries: (f_cost, g_cost, col, row)
        # g_cost included so ties in f break on cheaper g (consistent tiebreak)
        open_list = []
        g_start   = 0.0
        f_start   = self.heuristic(start_col, start_row, goal_col, goal_row)
        heapq.heappush(open_list, (f_start, g_start, start_col, start_row))

        # g_cost map — (col, row) → best known cost from start
        g_cost = {(start_col, start_row): 0.0}

        # Closed set — cells already fully expanded
        closed_set = set()

        # Parent map — (col, row) → (parent_col, parent_row) or None for start
        parent = {(start_col, start_row): None}

        nodes_expanded = 0

        # V2.1: track explored cells for visualization only (no logic impact)
        self._astar_explored_cells = []

        while open_list:
            f, g, col, row = heapq.heappop(open_list)
            cell = (col, row)

            # Discard stale heap entries (a cheaper path was found later)
            if cell in closed_set:
                continue
            closed_set.add(cell)
            nodes_expanded += 1
            self._astar_explored_cells.append(cell)   # V2.1: visualization only

            # Goal reached — reconstruct and return the path
            if col == goal_col and row == goal_row:
                path = self.reconstruct_path(parent, goal_col, goal_row)
                return path, nodes_expanded

            for move_cost, nc, nr in self.get_neighbors(col, row):
                neighbour = (nc, nr)
                if neighbour in closed_set:
                    continue

                tentative_g = g + move_cost

                if tentative_g < g_cost.get(neighbour, float('inf')):
                    g_cost[neighbour] = tentative_g
                    f_new = tentative_g + self.heuristic(nc, nr, goal_col, goal_row)
                    heapq.heappush(open_list, (f_new, tentative_g, nc, nr))
                    parent[neighbour] = cell

        # Open list exhausted — no path exists
        return [], nodes_expanded

    def reconstruct_path(self, parent: dict,
                         goal_col: int, goal_row: int) -> list:
        """
        Walk the parent dictionary backwards from the goal to rebuild the path.

        Parameters
        ----------
        parent             : dict mapping (col, row) → (parent_col, parent_row)
        goal_col, goal_row : the goal cell

        Returns
        -------
        list of (col, row) tuples ordered start → goal
        """
        path = []
        cell = (goal_col, goal_row)

        while cell is not None:
            path.append(cell)
            cell = parent[cell]

        path.reverse()
        return path

    def plan_path(self):
        """
        Compute a global A* path from the drone's current position to the
        fixed red-goal cell and store it in self.path.

        What this function does
        -----------------------
        1. Converts the drone's live world position to a start grid cell.
        2. Converts the fixed goal world position to a goal grid cell.
        3. Runs A* on self.occupancy_grid.
        4. Converts every grid cell in the result to world coordinates.
        5. Stores the world-frame waypoints in self.path.
        6. Sets self.path_ready = True on success.

        What this function does NOT do
        ------------------------------
        - Does not publish cmd_vel.
        - Does not modify the control loop.
        - Does not change drone movement.
        - Does not rebuild the occupancy grid.

        Contract (unchanged from V1 placeholder)
        -----------------------------------------
        - self.path             : list of (x, y) world-frame tuples, or []
        - self.path_ready       : True iff a valid path was found
        - self.current_waypoint : reset to 0
        """
        self.path             = []
        self.path_ready       = False
        self.current_waypoint = 0

        # ── Convert start (current drone position) to grid ────────────────────
        start_col, start_row = self.world_to_grid(self.current_x, self.current_y)
        if start_col is None:
            self.get_logger().warn(
                f'Start ({self.current_x:.2f}, {self.current_y:.2f}) outside grid — planning aborted')
            return

        # ── Convert fixed goal position to grid ───────────────────────────────
        goal_col, goal_row = self.world_to_grid(
            DroneNavigator.GOAL_WORLD_X,
            DroneNavigator.GOAL_WORLD_Y)
        if goal_col is None:
            self.get_logger().warn(
                f'Goal ({DroneNavigator.GOAL_WORLD_X}, {DroneNavigator.GOAL_WORLD_Y}) outside grid — planning aborted')
            return

        # ── Run A* search ─────────────────────────────────────────────────────
        self.get_logger().info('─' * 50)
        self.get_logger().info('Planning Global Path...')
        self.get_logger().info(f'  Start : cell ({start_col}, {start_row})')
        self.get_logger().info(f'  Goal  : cell ({goal_col}, {goal_row})')

        _plan_start_time = self.get_clock().now()   # V2.1: visualization timing only

        grid_path, nodes_expanded = self.astar(
            start_col, start_row, goal_col, goal_row)

        if not grid_path:
            self.get_logger().warn('No Path Found — falling back to reactive navigation')
            self.get_logger().info('─' * 50)
            # V2.1: visualization data — record failed attempt too
            self._astar_grid_path     = []
            self._astar_nodes_expanded = nodes_expanded
            self._astar_planning_ms    = (
                self.get_clock().now() - _plan_start_time).nanoseconds / 1e6
            return

        # ── Convert grid cells → world-frame waypoints ────────────────────────
        world_waypoints = []
        for col, row in grid_path:
            wx, wy = self.grid_to_world(col, row)
            if wx is not None:
                world_waypoints.append((wx, wy))

        self.path             = world_waypoints
        self.path_ready       = True
        self.current_waypoint = 0

        # V2.1: store data for the A* visualization window only
        self._astar_grid_path      = grid_path
        self._astar_nodes_expanded = nodes_expanded
        self._astar_planning_ms    = (
            self.get_clock().now() - _plan_start_time).nanoseconds / 1e6

        self.get_logger().info('Path Found')
        self.get_logger().info(f'  Nodes expanded : {nodes_expanded}  |  Cells : {len(grid_path)}  |  Waypoints : {len(world_waypoints)}')
        self.get_logger().info('─' * 50)

    def follow_waypoints(self) -> Twist:
        """
        Produce a cmd_vel Twist to drive toward self.path[self.current_waypoint].

        Behaviour
        ---------
        - Returns zero Twist immediately when no valid path exists.
        - Computes the world-frame vector from the drone to the current waypoint.
        - Uses proportional control with DISTANCE-BASED SPEED SCALING:
            far from waypoint  → approaches WP_MAX_SPEED (1.2 m/s)
            near the waypoint  → speed ramps down toward WP_MIN_SPEED (0.15 m/s)
          so the drone slows smoothly on approach instead of cruising at a flat
          speed and overshooting.
        - Applies an ACCELERATION LIMIT (WP_MAX_ACCEL) between consecutive
          published velocities so transitions — including the jump to a new
          waypoint's heading — are blended rather than instantaneous, removing
          the abrupt stop/start at each waypoint and any oscillation.
        - Advances self.current_waypoint when within WP_ACCEPT_RADIUS (0.30 m).
        - When the final waypoint is reached sets self.path_ready = False and
          returns zero Twist so _handle_v2_states() transitions to STATE_FIND_GOAL.
        - angular.z = 0 and linear.z = 0 at all times (no rotation, no altitude —
          altitude control is untouched and lives entirely outside this function).

        Coordinate frame
        ----------------
        self.path holds world-frame (x, y) positions.
        self.current_x / current_y are world-frame odometry readings.
        The drone is assumed to face world +X at all times (yaw not corrected here),
        so world-frame dx maps directly onto body linear.x and dy onto linear.y.
        """
        WP_MAX_SPEED      = 1.20   # m/s — top horizontal speed when far from waypoint
        WP_MIN_SPEED       = 0.15  # m/s — floor speed while still approaching (avoids stall/oscillation)
        WP_ACCEPT_RADIUS  = 0.30   # m  — waypoint considered reached inside this radius
        WP_SLOWDOWN_RADIUS = 1.20  # m  — distance at which speed begins ramping down
        WP_MAX_ACCEL       = 2.5   # m/s^2 — max change in speed per control tick (20 Hz)
        CONTROL_DT          = 0.05  # s — matches the 20 Hz control_loop timer period

        cmd = Twist()

        # ── Guard: no path available ──────────────────────────────────────────
        if not self.path_ready or not self.path:
            self._wp_prev_vx = 0.0
            self._wp_prev_vy = 0.0
            return cmd

        # ── Guard: all waypoints already consumed ─────────────────────────────
        if self.current_waypoint >= len(self.path):
            self.path_ready = False
            self._wp_prev_vx = 0.0
            self._wp_prev_vy = 0.0
            return cmd

        # ── Get target waypoint ───────────────────────────────────────────────
        wp_x, wp_y = self.path[self.current_waypoint]

        # ── Compute error vector (world frame) ────────────────────────────────
        dx = wp_x - self.current_x
        dy = wp_y - self.current_y
        dist = math.sqrt(dx * dx + dy * dy)

        # ── Advance waypoint index when within acceptance radius ──────────────
        if dist < WP_ACCEPT_RADIUS:
            self.current_waypoint += 1
            self.get_logger().debug(
                f'[WP] {self.current_waypoint - 1} reached → '
                f'{self.current_waypoint}/{len(self.path)}')

            # Final waypoint reached — path complete
            if self.current_waypoint >= len(self.path):
                self.get_logger().info('Following Waypoints... complete')
                self.path_ready = False
                self._wp_prev_vx = 0.0
                self._wp_prev_vy = 0.0
                return cmd

            # Recompute toward the newly selected waypoint — the acceleration
            # limiter below blends this heading change smoothly rather than
            # snapping to it, so there is no abrupt stop/start at the corner.
            wp_x, wp_y = self.path[self.current_waypoint]
            dx = wp_x - self.current_x
            dy = wp_y - self.current_y
            dist = math.sqrt(dx * dx + dy * dy)

        # ── Distance-based speed scaling ────────────────────────────────────────
        # Far away (dist >= WP_SLOWDOWN_RADIUS): full WP_MAX_SPEED.
        # Close in (dist -> 0): ramps linearly down to WP_MIN_SPEED so the
        # drone decelerates into the waypoint instead of overshooting.
        if dist >= WP_SLOWDOWN_RADIUS:
            target_speed = WP_MAX_SPEED
        else:
            ratio = dist / WP_SLOWDOWN_RADIUS   # 0..1
            target_speed = WP_MIN_SPEED + (WP_MAX_SPEED - WP_MIN_SPEED) * ratio

        # ── Build the desired velocity vector (direction = error, magnitude = target_speed)
        if dist > 1e-6:
            desired_vx = (dx / dist) * target_speed
            desired_vy = (dy / dist) * target_speed
        else:
            desired_vx = 0.0
            desired_vy = 0.0

        # ── Acceleration-limited blending ───────────────────────────────────────
        # Cap how much vx/vy can change this tick so velocity transitions
        # (including the corner between waypoints) are smooth, not abrupt,
        # and steady-state error near the waypoint can't cause oscillation.
        max_delta = WP_MAX_ACCEL * CONTROL_DT

        delta_vx = float(np.clip(desired_vx - self._wp_prev_vx, -max_delta, max_delta))
        delta_vy = float(np.clip(desired_vy - self._wp_prev_vy, -max_delta, max_delta))

        vx = self._wp_prev_vx + delta_vx
        vy = self._wp_prev_vy + delta_vy

        # Final safety clamp to the configured horizontal speed envelope
        speed = math.sqrt(vx * vx + vy * vy)
        if speed > WP_MAX_SPEED:
            scale = WP_MAX_SPEED / speed
            vx *= scale
            vy *= scale

        self._wp_prev_vx = vx
        self._wp_prev_vy = vy

        cmd.linear.x  = vx
        cmd.linear.y  = vy
        cmd.linear.z  = 0.0   # altitude control unchanged — handled elsewhere
        cmd.angular.z = 0.0

        self.get_logger().debug(
            f'[WP] {self.current_waypoint}/{len(self.path)} '
            f'dist={dist:.2f}m  target_v={target_speed:.2f}  vx={vx:.2f}  vy={vy:.2f}')

        return cmd

    def return_to_path(self) -> Twist:
        """
        Re-join the planned path after an obstacle detour.

        Behaviour
        ---------
        When self.returning_to_path is True:
          1. Searches self.path[self.current_waypoint:] for the waypoint
             whose world position is closest to the drone's current position.
             The search starts from self.current_waypoint (not index 0) so the
             drone always rejoins the path ahead of where it left it.
          2. Updates self.current_waypoint to that nearest index.
          3. Clears self.returning_to_path = False.
          4. Delegates immediately to follow_waypoints() so the drone resumes
             smooth proportional tracking without an extra control-loop tick.

        If the path is empty or exhausted the method clears the flag and
        returns a zero Twist; _handle_v2_states() will then transition to
        STATE_FIND_GOAL on the same tick.
        """
        cmd = Twist()

        if not self.returning_to_path:
            # Nothing to do — should not normally be called in this state
            return cmd

        # ── Guard: no valid path to return to ────────────────────────────────
        if not self.path_ready or not self.path:
            self.returning_to_path = False
            return cmd

        # ── Find nearest waypoint from current index onward ───────────────────
        # Clamp current_waypoint in case it drifted out of bounds during detour
        search_start = max(0, min(self.current_waypoint, len(self.path) - 1))

        best_idx  = search_start
        best_dist = float('inf')

        for idx in range(search_start, len(self.path)):
            wp_x, wp_y = self.path[idx]
            dx   = wp_x - self.current_x
            dy   = wp_y - self.current_y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < best_dist:
                best_dist = dist
                best_idx  = idx

        self.current_waypoint  = best_idx
        self.returning_to_path = False

        self.get_logger().info(
            f'Returning To Path... waypoint {best_idx}/{len(self.path) - 1} '
            f'({best_dist:.2f} m away)')

        # Delegate immediately to follow_waypoints so no tick is wasted
        return self.follow_waypoints()

    def _handle_v2_states(self) -> tuple:
        """
        Intercept V2 path-planning states before V1 obstacle / goal logic runs.

        Returns (handled: bool, cmd: Twist).
        When handled is True, control_loop() publishes cmd and returns immediately.
        When handled is False, control_loop() continues to V1 goal-alignment and
        reactive obstacle avoidance as normal.

        State machine
        -------------
        STATE_PLAN_PATH
            Run plan_path() exactly once per entry.
            → STATE_FOLLOW_PATH  if a valid path was produced
            → STATE_FIND_GOAL    if no path was found (V1 fallback)

        STATE_FOLLOW_PATH
            Call follow_waypoints() every tick.
            Obstacle intercept: if any obstacle flag is set, suspend waypoint
            following, set self.returning_to_path = True, and transition to
            STATE_RETURN_TO_PATH so the drone rejoins the path after clearing.
            → STATE_RETURN_TO_PATH  when an obstacle is detected
            → STATE_FIND_GOAL       when self.path_ready becomes False
                                    (all waypoints consumed — hand off to V1)

        STATE_RETURN_TO_PATH
            Call return_to_path() which finds the nearest remaining waypoint,
            resets self.returning_to_path = False, and delegates to
            follow_waypoints() in the same tick.
            → STATE_FOLLOW_PATH  once return_to_path() clears the flag
            → STATE_FIND_GOAL    if the path has been exhausted
        """
        # ── PLAN_PATH — run planner once, then advance ────────────────────────
        if self.state == STATE_PLAN_PATH:
            self.plan_path()
            if self.path_ready:
                self.state = STATE_FOLLOW_PATH
                self.get_logger().info('Following Waypoints...')
            else:
                self.state = STATE_FIND_GOAL
                self.get_logger().warn('No Path Found — reactive navigation active')
            return True, Twist()

        # ── FOLLOW_PATH — proportional waypoint following with obstacle intercept
        if self.state == STATE_FOLLOW_PATH:
            # Obstacle detected while following path → suspend and rejoin later
            if self.obstacle_left or self.obstacle_center or self.obstacle_right:
                self.returning_to_path = True
                self.state = STATE_RETURN_TO_PATH
                self.get_logger().info(
                    f'Obstacle Detected  '
                    f'[L:{int(self.obstacle_left)} '
                    f'C:{int(self.obstacle_center)} '
                    f'R:{int(self.obstacle_right)}]'
                    f' — suspending waypoint following')
                # Return zero Twist this tick; V1 avoidance takes over next tick
                # via return_to_path → follow_waypoints handoff
                return True, Twist()

            cmd = self.follow_waypoints()

            # Path exhausted — hand off to V1 goal detection / alignment
            if not self.path_ready or self.current_waypoint >= len(self.path):
                self.state = STATE_FIND_GOAL
                self.get_logger().info('Waypoints complete — searching for goal')
            return True, cmd

        # ── RETURN_TO_PATH — find nearest waypoint and resume following ────────
        if self.state == STATE_RETURN_TO_PATH:
            cmd = self.return_to_path()
            # return_to_path() clears self.returning_to_path when done
            if not self.returning_to_path:
                if self.path_ready and self.current_waypoint < len(self.path):
                    self.state = STATE_FOLLOW_PATH
                    self.get_logger().info('Following Waypoints... (resumed)')
                else:
                    self.state = STATE_FIND_GOAL
                    self.get_logger().info('Path exhausted after detour — searching for goal')
            return True, cmd

        # Not a V2 state — let control_loop() handle it
        return False, Twist()


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