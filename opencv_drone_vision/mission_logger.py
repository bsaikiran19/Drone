#!/usr/bin/env python3
"""
mission_logger.py

Lightweight, non-invasive CSV telemetry logger for
opencv_navigation_v4.DroneNavigator.

Contract
--------
- READ-ONLY with respect to DroneNavigator: this class never sets, mutates,
  or overrides any attribute on the `node` it is given. It only calls
  `getattr(node, ...)` and one pre-existing, side-effect-free method
  (`node.nav_core.distance_to_goal()`).
- Never touches navigation decisions: nothing here is consulted by
  `control_loop()`, `_yolo_inference_loop()`, or `NavigationCore` — data
  flows one way, out of the node and into the CSV.
- Any bookkeeping needed across ticks (e.g. previous Kalman predictions,
  for computing prediction error) is kept as internal state on the
  `MissionLogger` instance itself, never written back onto `node`.
- No value is estimated or invented. Where a metric cannot yet be computed
  (e.g. no YOLO inference has run yet, or a track has no prior prediction
  to compare against), the corresponding CSV cell is left blank.

Usage (see opencv_navigation_v4.py for the exact three call sites):
    self.mission_logger = MissionLogger(self)   # in __init__
    self.mission_logger.log()                   # once per control_loop tick
"""

import atexit
import csv
import os
import time
from datetime import datetime


CSV_COLUMNS = [
    'Timestamp',
    'Mission_Time',
    'Drone_X',
    'Drone_Y',
    'Drone_Z',
    'Yaw',
    'Distance_To_Goal',
    'Linear_Velocity_X',
    'Linear_Velocity_Y',
    'Linear_Velocity_Z',
    'Angular_Velocity_Z',
    'Current_State',
    'Navigation_Decision',
    'Risk_Level',
    'Mission_Status',
    'Goal_Reached',
    'Obstacle_Left',
    'Obstacle_Center',
    'Obstacle_Right',
    'Goal_Visible',
    'Goal_Centered',
    'Goal_Area',
    'Detection_Count',
    'Tracked_Object_Count',
    'Object_Classes',
    'Prediction_Count',
    'Kalman_Prediction_Error',
    'YOLO_Inference_Time_ms',
    'YOLO_FPS',
]


class MissionLogger:
    """Appends one CSV row per call to `log()`. Intended to be called once
    per `control_loop()` tick (20 Hz / every 50 ms), from three call sites
    so the hover, takeoff, and NavigationCore branches are all covered —
    see the integration notes for the exact insertion points.
    """

    def __init__(self, node, log_dir='logs', flush_every=20):
        """
        Args:
            node: The DroneNavigator instance to read telemetry from.
            log_dir: Directory (created if missing) the CSV is written
                into. Relative paths are resolved against the current
                working directory the node process was launched from.
            flush_every: Number of rows between disk flushes. Buffered
                writing — rows are appended to an in-memory-buffered file
                handle and only forced to disk every `flush_every` rows
                (default: 20 rows ≈ 1 second at 20 Hz), so per-tick logging
                overhead stays a single `writerow()` call with no per-row
                disk I/O.
        """
        self.node = node
        self._flush_every = flush_every
        self._rows_since_flush = 0

        os.makedirs(log_dir, exist_ok=True)
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = os.path.join(log_dir, f'mission_log_{run_id}.csv')

        # Buffered file handle (OS/library-level buffering); we control the
        # flush cadence explicitly via flush_every rather than flushing on
        # every write, so logging cannot become a bottleneck for the 20 Hz
        # control loop.
        self._file = open(self.csv_path, 'w', newline='', buffering=1)
        self._writer = csv.writer(self._file)
        self._writer.writerow(CSV_COLUMNS)

        self._start_time = time.time()

        # Internal-only bookkeeping for Kalman prediction error. Keyed by
        # track_id -> (predicted_x, predicted_y) from the last YOLO tick
        # that track_id was seen. Never written onto `node`.
        self._prev_predictions = {}
        # Identity of the last `node.latest_detections` list object we
        # processed. latest_detections is reassigned to a brand-new list
        # object every YOLO inference tick (~8 Hz) but control_loop() calls
        # log() at ~20 Hz, so this lets us detect "this is the same set of
        # detections as last tick, don't reprocess" without touching node.
        self._last_detections_identity = None
        self._last_mean_kalman_error = ''  # cached until next new YOLO tick

        node.get_logger().info(f'[MissionLogger] logging to {self.csv_path}')

        # Safety net: flush + close even if destroy_node()/shutdown paths
        # are skipped (e.g. Ctrl+C during shutdown). Registered on the
        # logger itself — does not require any change to main() or
        # DroneNavigator's shutdown sequence.
        atexit.register(self.close)

    # ------------------------------------------------------------------
    def _object_classes(self, detections):
        if not detections:
            return ''
        return ';'.join(sorted({d['class_name'] for d in detections}))

    def _prediction_count(self, detections):
        # Every entry in latest_detections already carries predicted_x/
        # predicted_y (assigned in _yolo_inference_loop before the dict is
        # appended), so this is a direct count, not an estimate.
        return sum(1 for d in detections if 'predicted_x' in d and 'predicted_y' in d)

    def _kalman_prediction_error(self, detections):
        """Mean pixel-distance between each track's PREVIOUSLY predicted
        position and its ACTUAL measured position this tick, averaged over
        all tracks for which a prior prediction exists.

        Only recomputed when `detections` is a new YOLO tick's list (see
        `_last_detections_identity` above) — otherwise the cached value
        from the last new tick is returned unchanged, so repeated log()
        calls between YOLO ticks don't corrupt the per-track history.

        Returns '' (NA) until at least one track has been seen on two
        consecutive YOLO ticks, since there is no prior prediction to
        compare against before that.
        """
        if not detections:
            return self._last_mean_kalman_error

        current_identity = id(detections)
        if current_identity == self._last_detections_identity:
            return self._last_mean_kalman_error

        errors = []
        for det in detections:
            track_id = det['track_id']
            prev = self._prev_predictions.get(track_id)
            if prev is not None:
                prev_x, prev_y = prev
                error_px = ((det['center_x'] - prev_x) ** 2 +
                            (det['center_y'] - prev_y) ** 2) ** 0.5
                errors.append(error_px)
            self._prev_predictions[track_id] = (det['predicted_x'], det['predicted_y'])

        self._last_detections_identity = current_identity
        self._last_mean_kalman_error = f'{(sum(errors) / len(errors)):.2f}' if errors else ''
        return self._last_mean_kalman_error

    # ------------------------------------------------------------------
    def log(self):
        """Reads current DroneNavigator state and appends one CSV row.
        Never raises into the caller's control flow on a bad/missing
        attribute — falls back to blank ('') for that single field so one
        unexpected None can't take down the control loop.
        """
        node = self.node
        now_wall = datetime.now().isoformat()
        mission_time = time.time() - self._start_time

        try:
            distance_to_goal = f'{node.nav_core.distance_to_goal():.4f}'
        except Exception:
            distance_to_goal = ''

        detections = getattr(node, 'latest_detections', []) or []

        inference_ms = getattr(node, 'last_inference_time_ms', None)
        inference_fps = getattr(node, 'last_inference_fps', None)

        row = [
            now_wall,
            f'{mission_time:.3f}',
            f'{getattr(node, "current_x", 0.0):.4f}',
            f'{getattr(node, "current_y", 0.0):.4f}',
            f'{getattr(node, "altitude", 0.0):.4f}',
            f'{getattr(node, "current_yaw", 0.0):.4f}',
            distance_to_goal,
            f'{getattr(node, "_log_linear_x", 0.0):.4f}',
            f'{getattr(node, "_log_linear_y", 0.0):.4f}',
            f'{getattr(node, "_log_linear_z", 0.0):.4f}',
            f'{getattr(node, "_log_angular_z", 0.0):.4f}',
            getattr(node, 'state', ''),
            getattr(node, 'navigation_decision', ''),
            getattr(node, 'current_risk_level', ''),
            'SUCCESS' if getattr(node, 'mission_complete', False) else 'RUNNING',
            getattr(node, 'mission_complete', False),
            getattr(node, 'obstacle_left', ''),
            getattr(node, 'obstacle_center', ''),
            getattr(node, 'obstacle_right', ''),
            getattr(node, 'goal_visible', ''),
            getattr(node, 'goal_centered', ''),
            f'{getattr(node, "goal_area", 0.0):.2f}',
            len(detections),
            len(getattr(node, 'kalman_tracks', {})),
            self._object_classes(detections),
            self._prediction_count(detections),
            self._kalman_prediction_error(detections),
            f'{inference_ms:.2f}' if inference_ms is not None else '',
            f'{inference_fps:.2f}' if inference_fps is not None else '',
        ]

        self._writer.writerow(row)
        self._rows_since_flush += 1
        if self._rows_since_flush >= self._flush_every:
            self._file.flush()
            self._rows_since_flush = 0

    # ------------------------------------------------------------------
    def close(self):
        """Flushes and closes the CSV file. Safe to call more than once
        (e.g. once explicitly and once via atexit)."""
        if self._file and not self._file.closed:
            try:
                self._file.flush()
            finally:
                self._file.close()
