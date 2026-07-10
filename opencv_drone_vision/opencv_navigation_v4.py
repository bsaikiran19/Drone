#!/usr/bin/env python3
"""
Version 4 — Phase 1 flight control + Phase 2 YOLOv8 perception.

NavigationCore is the ONLY source of navigation decisions. This node's job
is strictly: receive sensor data -> update NavigationCore -> receive a
NavigationCommand -> publish Twist. No movement logic lives outside
NavigationCore.

All of the old reactive obstacle-avoidance / goal-alignment / precision-
landing state machine has been removed (see "REMOVED / OBSOLETE" note
below). Camera perception (obstacle + goal detection, and now YOLOv8 object
detection) is kept and still runs every frame, but it only updates state
for logging/debug — it no longer drives the drone. It exists so future
phases can feed obstacle, goal, and object-detection information into
NavigationCore.

Phase 2 adds YOLOv8 object detection on the front camera feed. The camera
callback only stores the latest frame (lightweight, 20 Hz control loop is
never blocked); a dedicated timer runs YOLO inference at a lower, configurable
rate and stores results in `self.latest_detections`. The model is loaded from
this package's share directory (portable, independent of CWD) with a
try/except guard — if loading fails, YOLO is disabled and navigation
continues unaffected. Phase 3 adds multi-object tracking, Phase 4 adds
per-track Kalman motion prediction, and Phase 5 adds a rule-based Predictive
Dynamic Obstacle Avoidance decision engine on top of that — see
`_update_navigation_decision`. All of it is detection/recommendation only:
nothing in this file publishes a Twist derived from it, and NavigationCore
is untouched. A future phase can have NavigationCore consume
`self.navigation_decision` directly with no architectural change here.
"""

import os

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from ament_index_python.packages import get_package_share_directory

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String

import cv2
import numpy as np
from cv_bridge import CvBridge
import math
import time

from ultralytics import YOLO

# NavigationCore is now the only source of navigation decisions.
from .navigation_core import NavigationCore


# =============================================================================
# CONFIGURATION
# =============================================================================

# ─── HSV colour bounds (perception) ──────────────────────────────────────────

RED_LOWER1 = np.array([0,   120,  70])
RED_UPPER1 = np.array([10,  255, 255])
RED_LOWER2 = np.array([170, 120,  70])
RED_UPPER2 = np.array([180, 255, 255])

OBSTACLE_LOWER = np.array([0,   0,   0])
OBSTACLE_UPPER = np.array([180, 255,  80])

# ─── Navigation states ────────────────────────────────────────────────────────
# Only the states actually driven by control_loop() remain. The old reactive
# obstacle-avoidance / goal-alignment / precision-landing states have been
# removed — see "REMOVED / OBSOLETE NAVIGATION LOGIC" below.

STATE_TAKEOFF      = 'TAKEOFF'
STATE_FIND_GOAL     = 'FIND_GOAL'     # NavigationCore is actively flying to the goal
STATE_MISSION_DONE  = 'MISSION_DONE'  # goal reached, drone is hovering

# ─── Obstacle zone thresholds (fraction of image width) — perception only ────

ZONE_LEFT_MAX   = 0.35
ZONE_RIGHT_MIN  = 0.65

OBSTACLE_MIN_AREA = 3000

# ─── Goal marker detection thresholds — perception only ──────────────────────

GOAL_MIN_AREA   = 1500
GOAL_CENTER_TOL = 0.15

VERT_SPEED = 0.50   # m/s — takeoff climb rate

# ─── YOLOv8 object detection (perception only) ───────────────────────────────
# Phase 2. Runs on the front camera feed for visual object detection only.
# Detections are drawn/displayed for debug purposes and are NOT read by
# control_loop() or passed into NavigationCore. They have zero effect on
# UAV motion in this phase.
#
# Model is resolved at runtime from this package's share directory (see
# _resolve_yolo_model_path()), NOT a hardcoded/CWD-relative path, so the
# node works regardless of where `ros2 run` is launched from.
YOLO_PACKAGE_NAME     = 'opencv_drone_vision'
YOLO_MODEL_RELPATH    = os.path.join('models', 'yolov8n.pt')
YOLO_CONFIDENCE_MIN   = 0.5
YOLO_INFERENCE_HZ     = 8.0   # dedicated inference timer, independent of the
                               # 20 Hz control loop and the camera callback

# ─── Kalman motion prediction (perception only, Phase 4) ────────────────────
# One constant-velocity Kalman filter per track ID, keyed off the ByteTrack
# IDs from Phase 3. Predictions are cached in self.latest_detections for
# display/debug and future phases (e.g. a Phase 5 decision engine) — they are
# NOT read by control_loop() or NavigationCore and have zero effect on UAV
# motion in this phase.
KALMAN_TRACK_TIMEOUT_SEC = 1.5   # seconds a track's filter survives with no
                                  # matching detection before being discarded

# ─── Predictive Dynamic Obstacle Avoidance — Decision Engine (Phase 5) ──────
# Rule-based, RECOMMENDATION ONLY. Reads each tracked object's predicted
# next position (Phase 4 Kalman output) plus its bounding-box size (a
# proximity proxy — this is a monocular camera, so there is no real depth)
# to classify risk and recommend an action. This only ever writes
# self.navigation_decision / self.current_risk_level / self.decision_object.
# It never publishes a Twist, never touches NavigationCore, and never
# overrides control_loop(). Designed so a future phase can have
# NavigationCore consume self.navigation_decision directly.

# Collision zone — tight band directly in front of the UAV (normalized
# image-fraction coordinates, X = horizontal, Y = vertical). A predicted
# position landing in here on an object large enough to be "close" is
# treated as an imminent collision.
COLLISION_ZONE_X_FRAC   = (0.35, 0.65)
COLLISION_ZONE_Y_FRAC   = (0.15, 0.85)
COLLISION_MIN_AREA_FRAC = 0.05   # bbox area / frame area — "close" threshold

# Warning zone — wider band that catches objects on a converging path
# before they reach the collision zone.
WARNING_ZONE_X_FRAC   = (0.20, 0.80)
WARNING_ZONE_Y_FRAC   = (0.10, 0.90)
WARNING_MIN_AREA_FRAC = 0.015

# Risk level vocabulary.
RISK_SAFE      = 'SAFE'
RISK_WARNING   = 'WARNING'
RISK_COLLISION = 'COLLISION'
RISK_RANK      = {RISK_SAFE: 0, RISK_WARNING: 1, RISK_COLLISION: 2}
RISK_COLORS    = {
    RISK_SAFE:      (0, 255, 0),    # green  (BGR)
    RISK_WARNING:   (0, 255, 255),  # yellow (BGR)
    RISK_COLLISION: (0, 0, 255),    # red    (BGR)
}

# Decision engine output vocabulary — stored in self.navigation_decision.
DECISION_GO_TO_GOAL = 'GO_TO_GOAL'
DECISION_MOVE_LEFT  = 'MOVE_LEFT'
DECISION_MOVE_RIGHT = 'MOVE_RIGHT'
DECISION_ROTATE     = 'ROTATE'
DECISION_STOP       = 'STOP'

# ─── NavigationCore configuration ─────────────────────────────────────────────
# Phase 1 mission goal (world frame). NavigationCore flies straight to this
# point and hovers once it arrives — no path planning, no waypoints.
NAV_GOAL_X = 9.0
NAV_GOAL_Y = 0.0
NAV_GOAL_Z = 2.0

NAV_MAX_LINEAR_SPEED  = 1.0    # m/s
NAV_MAX_ANGULAR_SPEED = 1.0    # rad/s
NAV_GOAL_TOLERANCE    = 0.5    # m


# =============================================================================
# REMOVED / OBSOLETE NAVIGATION LOGIC (Phase 1)
# =============================================================================
# The following systems belonged to the pre-NavigationCore navigation stack
# and no longer run or influence UAV motion:
#   - Reactive obstacle-avoidance controller (_obstacle_action, _free_space_side)
#   - Goal alignment / goal-seeking yaw correction (_align_over_goal,
#     _goal_yaw_correction)
#   - Precision landing-pad alignment (precision_align)
#   - Controlled descent / stepped descent (controlled_descent,
#     _stepped_descent_command)
#   - Final soft landing / touchdown detection (final_landing)
#   - The AVOID_LEFT / AVOID_RIGHT / AVOID_BACK / ROTATE / ALIGN_GOAL /
#     PRECISION_ALIGN / CONTROLLED_DESCENT / FINAL_LANDING / MISSION_COMPLETE
#     states, and the old A*/occupancy-grid path planner and waypoint
#     follower they replaced
# These are not implemented anywhere in this file. If you need to reference
# the old implementation, pull it from version control history rather than
# re-adding it here — nothing in Phase 1 should execute it.
# =============================================================================


class _KalmanTrack:
    """One constant-velocity Kalman filter for a single tracked object.

    Perception-only helper for Phase 4. Instances are created, updated, and
    discarded entirely inside DroneNavigator's YOLO inference loop, keyed by
    the ByteTrack track ID from Phase 3. Nothing here is read by
    control_loop() or NavigationCore.

    State vector:  [x, y, vx, vy]^T  (pixel position + pixel velocity)
    Measurement:   [x, y]^T          (detected object center this tick)
    """

    def __init__(self, cx: float, cy: float, timestamp: float):
        kf = cv2.KalmanFilter(4, 2)
        # Constant-velocity model: x' = x + vx, y' = y + vy (dt folded into
        # the per-tick process noise below rather than tracked explicitly).
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 1e-2
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        kf.errorCovPost        = np.eye(4, dtype=np.float32)

        # Seed the filter directly at the first measurement, zero velocity.
        kf.statePost = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float32)
        # Prime statePre from statePost via the transition matrix so the
        # *next* call's correct() has a valid prior to work from.
        kf.predict()

        self.kf        = kf
        self.last_seen = timestamp

    def update(self, cx: float, cy: float, timestamp: float):
        """Correct with this tick's measurement, then predict the next
        position. Returns (predicted_x, predicted_y) as floats."""
        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
        self.kf.correct(measurement)
        prediction = self.kf.predict()
        self.last_seen = timestamp
        return float(prediction[0, 0]), float(prediction[1, 0])


class DroneNavigator(Node):
    """Camera-based perception + NavigationCore-driven flight for a UAV."""

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self):
        super().__init__('drone_navigator')

        self.bridge = CvBridge()

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ── ROS Interfaces: Subscribers ────────────────────────────────────────
        # Subscriber 1: front camera — obstacle detection ONLY
        self.create_subscription(
            Image, '/drone/front_camera/image_raw',
            self.front_camera_callback, qos_sensor)

        # Subscriber 2: down camera — goal detection ONLY
        self.create_subscription(
            Image, '/drone/down_camera/image_raw',
            self.down_camera_callback, qos_sensor)

        self.create_subscription(
            Odometry, '/drone/odom',
            self.odom_callback, 10)

        # ── ROS Interfaces: Publishers ───────────────────────────────────────
        self.cmd_pub    = self.create_publisher(Twist,  '/drone/cmd_vel', 10)
        self.enable_pub = self.create_publisher(Bool,   '/drone/enable',  10)
        self.status_pub = self.create_publisher(String, '/drone/mission_status', 10)

        # ── State: mission / pose ────────────────────────────────────────────
        self.state       = STATE_TAKEOFF
        self.altitude    = 0.0
        self.current_x   = 0.0
        self.current_y   = 0.0
        self.current_yaw = 0.0

        self.takeoff_ticks           = 0
        self.mission_complete        = False
        self.mission_complete_logged = False

        # ── State: obstacle perception (set only by front camera) ───────────
        self.obstacle_left   = False
        self.obstacle_center = False
        self.obstacle_right  = False

        # ── State: goal perception (set only by down camera) ────────────────
        self.goal_visible  = False
        self.goal_centered = False
        self.goal_area     = 0.0
        self.goal_cx        = 0
        self.goal_cy        = 0

        # Front camera frame dimensions
        self.front_img_width  = 640
        self.front_img_height = 480

        # Down camera frame dimensions
        self.down_img_width  = 640
        self.down_img_height = 480

        # ── YOLOv8 (Phase 3, perception only) ────────────────────────────────
        # Loaded once at startup and reused for every inference tick.
        # Detections/tracks from this model are for display/debug only — they
        # are never read by control_loop() and never reach NavigationCore.
        #
        # Storage for future phases (Kalman / prediction / decision engine).
        # Each entry carries a stable track_id + center + timestamp so Phase 4
        # can consume it directly. Not consumed anywhere yet.
        self.latest_detections = []

        # Preferred tracker for Ultralytics' built-in tracking API. ByteTrack
        # is tried first; if its config can't be loaded for any reason,
        # _yolo_inference_loop() permanently falls back to Ultralytics'
        # default tracker (BoT-SORT) rather than erroring out.
        self.tracker_config = 'bytetrack.yaml'

        # Phase 4 — one _KalmanTrack per active track_id, keyed by that ID.
        # Created the first time a track_id is seen, corrected/predicted
        # every tick it reappears, and dropped after KALMAN_TRACK_TIMEOUT_SEC
        # of absence (see _purge_stale_kalman_tracks()). Perception-only —
        # never read by control_loop() or NavigationCore.
        self.kalman_tracks = {}

        # Phase 5 — Predictive Dynamic Obstacle Avoidance decision engine.
        # Recomputed every YOLO inference tick by _update_navigation_decision().
        # RECOMMENDATION ONLY: nobody reads these yet — control_loop() and
        # NavigationCore are completely untouched. Kept as a flat, simple
        # attribute so a future phase can have NavigationCore consume
        # self.navigation_decision directly with no architectural change.
        self.navigation_decision = DECISION_GO_TO_GOAL
        self.current_risk_level  = RISK_SAFE
        self.decision_object     = None  # detection dict driving the
                                          # current decision, or None

        # Latest raw front-camera frame, updated by front_camera_callback and
        # consumed by the YOLO inference timer. Keeps the camera callback
        # itself lightweight — no inference runs inside it.
        self.latest_front_frame = None

        self.yolo_enabled = False
        self.yolo_model = None
        try:
            model_path = self._resolve_yolo_model_path()
            self.get_logger().info(f'Loading YOLOv8 model: {model_path}')
            self.yolo_model = YOLO(model_path)
            self.yolo_enabled = True
            self.get_logger().info('YOLOv8 model loaded')
        except Exception as e:
            self.yolo_enabled = False
            self.yolo_model = None
            self.get_logger().error(
                f'YOLOv8 model failed to load ({e}). '
                f'Continuing WITHOUT object detection — navigation is unaffected.'
            )

        # ── NavigationCore: the only source of navigation decisions ─────────
        self.nav_core = NavigationCore(
            max_linear_speed=NAV_MAX_LINEAR_SPEED,
            max_angular_speed=NAV_MAX_ANGULAR_SPEED,
            goal_tolerance=NAV_GOAL_TOLERANCE,
        )
        self.nav_core.set_goal(NAV_GOAL_X, NAV_GOAL_Y, NAV_GOAL_Z)

        # ── Timers ────────────────────────────────────────────────────────────
        # Control loop at 20 Hz — flight behaviour is untouched by YOLO.
        self.create_timer(0.05, self.control_loop)

        # YOLOv8 inference timer — runs independently at YOLO_INFERENCE_HZ
        # (default 8 Hz), decoupled from both the 20 Hz control loop and the
        # camera callback rate. Only created if the model loaded successfully.
        if self.yolo_enabled:
            self.create_timer(1.0 / YOLO_INFERENCE_HZ, self._yolo_inference_loop)

        self.get_logger().info('=' * 50)
        self.get_logger().info('MISSION STARTED')
        self.get_logger().info('=' * 50)

    # =========================================================================
    # YOLOv8 MODEL PATH RESOLUTION
    # =========================================================================

    def _resolve_yolo_model_path(self) -> str:
        """Resolves the YOLOv8 weights path via the package share directory.

        This makes model loading portable: it does not depend on the
        current working directory the node happens to be launched from.
        Expected layout (installed by this package's setup):

            <share>/opencv_drone_vision/models/yolov8n.pt

        Returns:
            Absolute path to the YOLOv8 weights file.

        Raises:
            Exception: if the package share directory or the weights file
                cannot be located. Caught by the caller in `__init__`, which
                disables YOLO gracefully rather than crashing the node.
        """
        share_dir = get_package_share_directory(YOLO_PACKAGE_NAME)
        model_path = os.path.join(share_dir, YOLO_MODEL_RELPATH)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f'YOLO weights not found at {model_path}')
        return model_path

    # =========================================================================
    # CAMERA CALLBACKS
    # =========================================================================
    # Both callbacks only process images and update perception state. Neither
    # commands UAV movement directly. In future phases the obstacle/goal
    # state they populate will be passed into NavigationCore instead of being
    # used for logging/debug only.

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

        # Store the latest frame only — keep this callback lightweight.
        # Actual YOLO inference runs from a dedicated timer (see
        # _yolo_inference_loop) at YOLO_INFERENCE_HZ, decoupled from the
        # camera publish rate and from the 20 Hz control loop.
        self.latest_front_frame = frame

        # PLACEHOLDER — Static Obstacle Avoidance
        # Future phase: pass self.obstacle_left/center/right (or richer
        # contour data) into NavigationCore so it can factor obstacles into
        # its velocity command.

        # PLACEHOLDER — Dynamic Obstacle Tracking
        # Future phase: track moving obstacles across frames and feed
        # predicted positions into NavigationCore.

        # NOTE — YOLO Detection (Phase 2, this file)
        # Object detection runs from a dedicated timer (_yolo_inference_loop),
        # not from this callback — see the frame-store line above. It is
        # perception-only: detections are drawn/displayed and cached in
        # self.latest_detections, but never fed into NavigationCore. Tracking,
        # Kalman filtering, motion prediction, and obstacle-avoidance
        # decisions from these detections remain future-phase work.

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

        # PLACEHOLDER — Landing
        # Future phase: detect a landing marker/pad on this frame and pass
        # its position/alignment into NavigationCore for a landing approach.

        # PLACEHOLDER — Motion Prediction
        # Future phase: predict goal/pad motion between frames and feed the
        # prediction into NavigationCore.

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
    # Perception — YOLOv8 multi-object tracking (front camera only, Phase 3)
    # ─────────────────────────────────────────────────────────────────────────
    # Tracking + motion prediction only. This method does not set any state
    # that control_loop() or NavigationCore reads — it purely draws,
    # displays, and caches tracked objects (plus their predicted next
    # position) for future phases. Obstacle/collision avoidance, path
    # replanning, and velocity modification built on top of these
    # predictions are explicitly out of scope for this phase.

    def _yolo_inference_loop(self):
        """Runs Ultralytics multi-object tracking + per-track Kalman
        motion prediction on the latest front frame.

        Fired by its own timer at `YOLO_INFERENCE_HZ`, independent of the
        camera publish rate and the 20 Hz control loop. Operates on
        `self.latest_front_frame`, the most recent frame stored by
        `front_camera_callback`.

        Uses YOLOv8's built-in `.track()` API with `persist=True` so the
        tracker's internal state (and therefore object IDs) is carried over
        between calls on this same model instance, instead of being
        reinitialised every tick. ByteTrack is used by default; if its
        tracker config can't be loaded, this permanently falls back to
        Ultralytics' default tracker on the next call.

        For every tracked object above `YOLO_CONFIDENCE_MIN` that has been
        assigned a stable track ID:
          - a per-track_id `_KalmanTrack` is created the first time that ID
            appears, or corrected + advanced one step if it already exists
            (see `_KalmanTrack.update`);
          - a bounding box, class-name/track-ID/confidence caption, current
            center, and predicted next center (with a connecting line) are
            drawn on a copy of the frame and shown in its own OpenCV window;
          - a plain-dict summary — including `predicted_x`/`predicted_y` — is
            stored in `self.latest_detections` for future phases (e.g. a
            Phase 5 decision engine).

        Track IDs that stop appearing keep their Kalman filter alive for
        `KALMAN_TRACK_TIMEOUT_SEC` before being discarded — see
        `_purge_stale_kalman_tracks`.

        Nothing here is read by `control_loop()` or `NavigationCore`. If the
        model failed to load, or no frame has arrived yet, this is a no-op —
        navigation is completely unaffected either way.
        """
        if not self.yolo_enabled or self.yolo_model is None:
            return
        if self.latest_front_frame is None:
            return

        frame = self.latest_front_frame

        try:
            results = self.yolo_model.track(
                frame,
                persist=True,
                tracker=self.tracker_config,
                verbose=False,
            )[0]
        except Exception as e:
            if self.tracker_config is not None:
                # ByteTrack config unavailable — fall back to Ultralytics'
                # default tracker permanently and retry this tick once.
                self.get_logger().warn(
                    f'Tracker "{self.tracker_config}" unavailable ({e}); '
                    f'falling back to Ultralytics default tracker.'
                )
                self.tracker_config = None
                try:
                    results = self.yolo_model.track(
                        frame, persist=True, verbose=False)[0]
                except Exception as e2:
                    self.get_logger().error(f'YOLOv8 tracking error: {e2}')
                    return
            else:
                self.get_logger().error(f'YOLOv8 tracking error: {e}')
                return

        yolo_debug = frame.copy()
        detections = []

        for box in results.boxes:
            confidence = float(box.conf[0])
            if confidence < YOLO_CONFIDENCE_MIN:
                continue

            # Skip boxes the tracker hasn't associated with a stable ID yet
            # (e.g. first frame an object appears in). Only confirmed tracks
            # go into latest_detections / Phase 4 motion prediction.
            if box.id is None:
                continue
            track_id = int(box.id[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            class_name = self.yolo_model.names.get(class_id, str(class_id))
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            timestamp = time.time()

            # Phase 4 — one Kalman filter per track_id. New ID: create and
            # seed it (no velocity info yet, so predicted == current).
            # Existing ID: correct with this tick's measurement and predict
            # the next position.
            if track_id in self.kalman_tracks:
                predicted_x, predicted_y = self.kalman_tracks[track_id].update(
                    center_x, center_y, timestamp)
            else:
                self.kalman_tracks[track_id] = _KalmanTrack(
                    center_x, center_y, timestamp)
                predicted_x, predicted_y = float(center_x), float(center_y)

            detections.append({
                'track_id': track_id,
                'class_id': class_id,
                'class_name': class_name,
                'confidence': confidence,
                'bbox': (x1, y1, x2, y2),
                'center_x': center_x,
                'center_y': center_y,
                'predicted_x': predicted_x,
                'predicted_y': predicted_y,
                'timestamp': timestamp,
            })

            # Phase 5 — classify this object's collision risk from its
            # predicted position + bbox size, and tag it onto the detection
            # dict so _update_navigation_decision() below can aggregate
            # without reclassifying. Recommendation-only: this never
            # touches control_loop() or NavigationCore.
            risk_level = self._classify_risk(
                detections[-1], self.front_img_width, self.front_img_height)
            detections[-1]['risk_level'] = risk_level
            box_color = RISK_COLORS[risk_level]

            cv2.rectangle(yolo_debug, (x1, y1), (x2, y2), box_color, 2)
            label = f'{class_name.capitalize()} #{track_id} {confidence:.2f} [{risk_level}]'
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(yolo_debug,
                          (x1, max(0, y1 - label_h - 8)),
                          (x1 + label_w + 4, y1),
                          box_color, -1)
            cv2.putText(yolo_debug, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
            cv2.circle(yolo_debug, (center_x, center_y), 3, (0, 0, 255), -1)

            # Predicted next position (Phase 4, visualization only) — a
            # marker in this object's risk color plus a line from the
            # current center to it — this is the "Predicted Obstacle"
            # required by Phase 5's visualization spec.
            pred_pt = (int(round(predicted_x)), int(round(predicted_y)))
            cv2.line(yolo_debug, (center_x, center_y), pred_pt,
                     box_color, 1)
            cv2.drawMarker(yolo_debug, pred_pt, box_color,
                           markerType=cv2.MARKER_TILTED_CROSS,
                           markerSize=10, thickness=2)

        # Phase 5 — aggregate the per-object risk levels already tagged
        # above into a single navigation_decision. RECOMMENDATION ONLY: only
        # writes self.navigation_decision / self.current_risk_level /
        # self.decision_object — control_loop() and NavigationCore are
        # completely untouched.
        self._update_navigation_decision(detections, self.front_img_width)

        # Cache for future phases (e.g. NavigationCore reads
        # predicted_x/predicted_y/risk_level directly from here next phase).
        self.latest_detections = detections

        # Drop Kalman filters for track IDs that haven't reappeared within
        # the configured timeout — keeps self.kalman_tracks from growing
        # unbounded as objects leave the frame for good.
        self._purge_stale_kalman_tracks(time.time())

        # Phase 5 — draw the collision/warning zones plus the current
        # decision + risk level on the same debug frame used for tracking.
        self._draw_decision_overlay(yolo_debug)

        cv2.putText(yolo_debug, f'Tracked objects: {len(detections)}',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        cv2.imshow('Front Camera - YOLOv8 Tracking', yolo_debug)
        cv2.waitKey(1)

    def _purge_stale_kalman_tracks(self, now: float):
        """Removes `_KalmanTrack` entries not updated within the configured
        timeout (Phase 4).

        Called once per inference tick from `_yolo_inference_loop`. A track
        ID that momentarily drops out of detection (occlusion, a missed
        frame, etc.) keeps its filter — and therefore its motion history —
        alive for `KALMAN_TRACK_TIMEOUT_SEC` before being discarded, so a
        brief reappearance doesn't restart prediction from zero velocity.
        Perception-only bookkeeping; never touches control_loop() or
        NavigationCore.
        """
        stale_ids = [
            track_id for track_id, track in self.kalman_tracks.items()
            if (now - track.last_seen) > KALMAN_TRACK_TIMEOUT_SEC
        ]
        for track_id in stale_ids:
            del self.kalman_tracks[track_id]

    # ─────────────────────────────────────────────────────────────────────────
    # Decision Engine — Predictive Dynamic Obstacle Avoidance (Phase 5)
    # ─────────────────────────────────────────────────────────────────────────
    # Rule-based, RECOMMENDATION ONLY. Consumes the Phase 4 Kalman
    # predictions already attached to each tracked detection. Writes
    # self.navigation_decision / self.current_risk_level / self.decision_object
    # and nothing else — no Twist is published from here, NavigationCore is
    # not touched, and control_loop() does not call into this code at all.

    def _classify_risk(self, detection: dict, frame_w: int, frame_h: int) -> str:
        """Classifies one tracked object's collision risk (Phase 5).

        Uses the object's *predicted* next position (not its current
        position) so the risk level reflects where it is headed, not just
        where it is right now. Since this is a single monocular camera with
        no depth sensing, bounding-box area (as a fraction of the frame) is
        used as a proximity proxy — a large box means the object is close.

        An object is:
          - COLLISION if its predicted position falls inside the tight
            `COLLISION_ZONE_*_FRAC` band AND its bbox area fraction is at
            least `COLLISION_MIN_AREA_FRAC` (i.e. close enough to matter).
          - WARNING if its predicted position falls inside the wider
            `WARNING_ZONE_*_FRAC` band AND its bbox area fraction is at
            least `WARNING_MIN_AREA_FRAC`.
          - SAFE otherwise.

        Args:
            detection: One entry from the `detections` list built in
                `_yolo_inference_loop` — must have `predicted_x`,
                `predicted_y`, and `bbox`.
            frame_w: Front camera frame width, in pixels.
            frame_h: Front camera frame height, in pixels.

        Returns:
            One of `RISK_SAFE`, `RISK_WARNING`, `RISK_COLLISION`.
        """
        if frame_w <= 0 or frame_h <= 0:
            return RISK_SAFE

        px, py = detection['predicted_x'], detection['predicted_y']
        x1, y1, x2, y2 = detection['bbox']
        bbox_area = max(0, x2 - x1) * max(0, y2 - y1)
        area_frac = bbox_area / float(frame_w * frame_h)

        norm_px = px / float(frame_w)
        norm_py = py / float(frame_h)

        in_collision_zone = (
            COLLISION_ZONE_X_FRAC[0] <= norm_px <= COLLISION_ZONE_X_FRAC[1]
            and COLLISION_ZONE_Y_FRAC[0] <= norm_py <= COLLISION_ZONE_Y_FRAC[1]
        )
        in_warning_zone = (
            WARNING_ZONE_X_FRAC[0] <= norm_px <= WARNING_ZONE_X_FRAC[1]
            and WARNING_ZONE_Y_FRAC[0] <= norm_py <= WARNING_ZONE_Y_FRAC[1]
        )

        if in_collision_zone and area_frac >= COLLISION_MIN_AREA_FRAC:
            return RISK_COLLISION
        if in_warning_zone and area_frac >= WARNING_MIN_AREA_FRAC:
            return RISK_WARNING
        return RISK_SAFE

    def _update_navigation_decision(self, detections: list, frame_w: int) -> None:
        """Aggregates per-object risk into one navigation recommendation.

        Each `detection` in `detections` is expected to already carry a
        `risk_level` key (set in `_yolo_inference_loop` via
        `_classify_risk`). This method does not reclassify anything — it
        just picks the single worst-risk object and applies the decision
        rules below:

            IF worst risk is COLLISION      → STOP
            ELSE IF worst risk is WARNING:
                IF that object's predicted X is LEFT of center   → MOVE_RIGHT
                ELSE IF predicted X is RIGHT of center           → MOVE_LEFT
                ELSE (predicted X is CENTER)                     → ROTATE
            ELSE (no object above SAFE)     → GO_TO_GOAL

        "Left"/"right"/"center" reuse the same `ZONE_LEFT_MAX` /
        `ZONE_RIGHT_MIN` fractions the existing obstacle-zone perception
        uses, so the decision engine's notion of "in front of the UAV"
        stays consistent with the rest of this file.

        This is a RECOMMENDATION ONLY. It updates `self.navigation_decision`,
        `self.current_risk_level`, and `self.decision_object` and returns
        nothing else — it never publishes a Twist, never calls into
        NavigationCore, and is never called from control_loop().

        Args:
            detections: This tick's tracked objects, each already tagged
                with a `risk_level` key.
            frame_w: Front camera frame width, in pixels — used to
                normalize the worst object's predicted X for the
                left/right/center decision.
        """
        worst_detection = None
        worst_risk = RISK_SAFE

        for det in detections:
            risk = det.get('risk_level', RISK_SAFE)
            if RISK_RANK[risk] > RISK_RANK[worst_risk]:
                worst_risk = risk
                worst_detection = det

        self.current_risk_level = worst_risk
        self.decision_object = worst_detection

        if worst_risk == RISK_COLLISION:
            self.navigation_decision = DECISION_STOP
            return

        if worst_risk == RISK_WARNING and worst_detection is not None:
            norm_px = worst_detection['predicted_x'] / float(frame_w) if frame_w else 0.5
            if norm_px < ZONE_LEFT_MAX:
                self.navigation_decision = DECISION_MOVE_RIGHT   # obstacle left → go right
            elif norm_px > ZONE_RIGHT_MIN:
                self.navigation_decision = DECISION_MOVE_LEFT    # obstacle right → go left
            else:
                self.navigation_decision = DECISION_ROTATE       # obstacle center
            return

        self.navigation_decision = DECISION_GO_TO_GOAL

    def _draw_decision_overlay(self, debug) -> None:
        """Draws the Phase 5 collision/warning zones and decision HUD.

        Purely visual — reads `self.navigation_decision`,
        `self.current_risk_level`, `self.front_img_width/height`, and the
        zone-fraction constants; writes nothing back to `self`.

        Args:
            debug: The YOLO tracking debug frame (BGR image), modified
                in place.
        """
        h, w = debug.shape[:2]

        def zone_px(x_frac, y_frac):
            return (
                int(w * x_frac[0]), int(h * y_frac[0]),
                int(w * x_frac[1]), int(h * y_frac[1]),
            )

        # Warning zone — thin outline.
        wx1, wy1, wx2, wy2 = zone_px(WARNING_ZONE_X_FRAC, WARNING_ZONE_Y_FRAC)
        cv2.rectangle(debug, (wx1, wy1), (wx2, wy2),
                      RISK_COLORS[RISK_WARNING], 1)

        # Collision zone — thicker outline, translucent fill so it reads as
        # "the danger zone" without hiding what's behind it.
        cx1, cy1, cx2, cy2 = zone_px(COLLISION_ZONE_X_FRAC, COLLISION_ZONE_Y_FRAC)
        overlay = debug.copy()
        cv2.rectangle(overlay, (cx1, cy1), (cx2, cy2),
                      RISK_COLORS[RISK_COLLISION], -1)
        cv2.addWeighted(overlay, 0.12, debug, 0.88, 0, debug)
        cv2.rectangle(debug, (cx1, cy1), (cx2, cy2),
                      RISK_COLORS[RISK_COLLISION], 2)
        cv2.putText(debug, 'COLLISION ZONE', (cx1 + 4, cy1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    RISK_COLORS[RISK_COLLISION], 1)

        # Decision + risk HUD, color-coded to the current risk level.
        hud_color = RISK_COLORS[self.current_risk_level]
        cv2.putText(debug, f'Decision: {self.navigation_decision}',
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, hud_color, 2)
        cv2.putText(debug, f'Risk: {self.current_risk_level}',
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, hud_color, 2)

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

        # HUD
        cv2.putText(debug, f'State: {self.state}',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(debug, f'Ticks: {self.takeoff_ticks}',
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

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
        cv2.putText(debug, f'Area: {self.goal_area:.0f}',
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

    # =========================================================================
    # ODOMETRY CALLBACK
    # =========================================================================

    def odom_callback(self, msg: Odometry):
        self.altitude  = msg.pose.pose.position.z
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    # =========================================================================
    # NAVIGATION
    # =========================================================================
    # NavigationCore is the ONLY source of navigation decisions. This method
    # does not compute any velocity itself — it takes off, hands current pose
    # to NavigationCore, publishes whatever Twist NavigationCore returns, and
    # hovers once NavigationCore reports the goal reached.
    #
    # Control loop:
    #   Takeoff -> Update current pose -> NavigationCore.compute_navigation()
    #   -> Publish Twist -> Hover when goal reached

    def control_loop(self):
        cmd = Twist()

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
            # Hover behaviour: goal reached — publish zero velocity and keep
            # spinning normally. Never transitions into landing behaviour.
            self.cmd_pub.publish(Twist())
            return

        # Timer-based takeoff — 60 ticks at 20 Hz = 3 seconds. Once complete,
        # control hands off to NavigationCore below.
        if self.state == STATE_TAKEOFF:
            cmd.linear.z = VERT_SPEED
            self.takeoff_ticks += 1
            if self.takeoff_ticks >= 60:
                self.state = STATE_FIND_GOAL
                self.get_logger().info('Takeoff complete → NavigationCore engaged')
                enable_msg = Bool()
                enable_msg.data = True
                self.enable_pub.publish(enable_msg)
            self.cmd_pub.publish(cmd)
            return

        # ── NavigationCore drives every remaining tick ──────────────────────
        self.nav_core.update_pose(
            x=self.current_x, y=self.current_y,
            yaw=self.current_yaw, z=self.altitude,
        )
        self.nav_core.set_navigation_decision(
            self.navigation_decision
        )
        command = self.nav_core.compute_navigation()

        cmd.linear.x  = command.linear_x
        cmd.linear.y  = command.linear_y
        cmd.linear.z  = command.linear_z
        cmd.angular.z = command.angular_z
        self.cmd_pub.publish(cmd)

        if command.goal_reached:
            self.state = STATE_MISSION_DONE
            self.mission_complete = True
            if not self.mission_complete_logged:
                self.get_logger().info('===================================')
                self.get_logger().info('GOAL REACHED — HOVERING')
                self.get_logger().info('===================================')
                self.mission_complete_logged = True


# =============================================================================
# MAIN FUNCTION
# =============================================================================

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