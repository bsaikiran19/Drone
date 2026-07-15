"""
navigation_core.py

Core navigation intelligence for Version 4 of the autonomous UAV
navigation system.

This module implements a minimal "fly-towards-goal" navigation strategy:

    Current Position
            |
            v
    Compute Direction to Goal
            |
            v
    Generate Velocity Command
            |
            v
    Fly Towards Goal
            |
            v
    Hover When Goal Reached

There is intentionally no path planning, no occupancy grid, no A*, and
no waypoint queue. The UAV always computes a direct heading and speed
towards the single stored goal position, based on its current pose.

This file contains only navigation math. It has no dependency on ROS2,
publishers, subscribers, OpenCV, or YOLO. It is meant to be imported
and driven by a separate ROS2 node.

The data model is intentionally 3D (x, y, z) so that future phases
(altitude control, obstacle avoidance, landing, etc.) can build on top
of this class without changing its shape. For Phase 1, only X/Y are
used to generate motion; Z is carried through the API and always
returns zero velocity until altitude control is implemented.
"""

import math
from dataclasses import dataclass
from enum import Enum, auto


class MissionState(Enum):
    """High-level state of the navigation mission.

    IDLE: No goal has been set, or the goal was cleared. The UAV
        should not move.
    GO_TO_GOAL: A goal is set and the UAV is outside tolerance, so it
        is actively flying towards it.
    GOAL_REACHED: The UAV is within tolerance of the goal and should
        hover in place.
    """

    IDLE = auto()
    GO_TO_GOAL = auto()
    GOAL_REACHED = auto()


@dataclass
class Pose:
    """UAV position and heading in the world frame.

    Attributes:
        x: X position, in meters.
        y: Y position, in meters.
        z: Z position (altitude), in meters.
        yaw: Heading angle about the Z axis, in radians.
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0


@dataclass
class Goal:
    """Target position for the current mission leg.

    Attributes:
        x: Goal X position, in meters.
        y: Goal Y position, in meters.
        z: Goal Z position (altitude), in meters.
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class NavigationCommand:
    """Velocity command produced by the navigation core.

    Attributes:
        linear_x: Forward linear velocity, in meters per second.
        linear_y: Lateral linear velocity, in meters per second.
        linear_z: Vertical linear velocity, in meters per second.
            Reserved for future altitude control; always 0.0 for now.
        angular_z: Angular velocity about the yaw axis, in radians
            per second.
        goal_reached: True if the UAV is within tolerance of the goal.
        distance_to_goal: Current straight-line distance to the goal,
            in meters.
    """

    linear_x: float = 0.0
    linear_y: float = 0.0
    linear_z: float = 0.0
    angular_z: float = 0.0
    goal_reached: bool = False
    distance_to_goal: float = 0.0


@dataclass
class _NavigationGeometry:
    """Internal snapshot of goal-relative geometry.

    Bundles the values every navigation calculation depends on so
    they are computed exactly once per `compute_navigation()` call.

    Attributes:
        dx: Goal X minus current X, in meters.
        dy: Goal Y minus current Y, in meters.
        distance: Euclidean (X/Y) distance to the goal, in meters.
        heading_to_goal: Absolute bearing to the goal, in radians.
        heading_error: Shortest angular difference between the
            current yaw and `heading_to_goal`, in radians.
    """

    dx: float
    dy: float
    distance: float
    heading_to_goal: float
    heading_error: float


# ============================================================================
# Version 5 architecture
# ============================================================================
#
# Everything below this banner and above `class NavigationCore` is new,
# additive Version 5 scaffolding. It is intentionally NOT wired into
# NavigationCore or compute_navigation() yet — see the module-level notes
# below. None of the Version 4 code above this banner has been modified.
#
# Version 5 introduces a perception -> world-model -> planning -> decision
# -> control pipeline that will eventually sit in front of NavigationCore's
# simple direct-to-goal controller:
#
#     Detection (per-frame perception)
#             |
#             v
#     ObjectManager (detection -> persistent Track3D)
#             |
#             v
#     MotionPredictor (Track3D -> future position estimate)
#             |
#             v
#     EnvironmentModel (Track3D + prediction -> ObstacleEntry map)
#             |
#             v
#     WaypointManager  <----  MissionManager (mission legs / goals)
#             |                       |
#             v                       v
#     PlannerEngine (EnvironmentModel + waypoints -> PlannerResult)
#             |
#             v
#     DecisionEngine (PlannerResult + EnvironmentModel -> DecisionResult)
#             |
#             v
#     VelocityController (DecisionResult -> NavigationCommand-shaped output)
#
# For this pass, these classes are self-contained and unintegrated: they do
# not call into NavigationCore, and NavigationCore does not call into them.
# Wiring them together (e.g. having DecisionEngine drive
# `NavigationCore.set_navigation_decision()`, or having PlannerEngine feed
# WaypointManager into `NavigationCore.set_goal()`) is explicitly out of
# scope per the current task and is left for a later integration pass.
# ============================================================================


@dataclass
class Detection:
    """A single raw perception detection for one video/sensor frame.

    This is the lowest-level perception primitive in the Version 5
    pipeline: one bounding box (or point detection) with a class label
    and confidence, as produced directly by a detector (e.g. YOLO) for
    a single frame, before any temporal association/tracking has been
    applied. `ObjectManager` consumes streams of `Detection` objects
    and turns them into persistent `Track3D` objects.

    Attributes:
        detection_id: Identifier for this detection within its source
            frame (e.g. detector output index). Not a persistent
            track identity — that is assigned by `ObjectManager`.
        label: Class label of the detected object (e.g. "person",
            "drone", "tree", "building").
        confidence: Detector confidence score, in the range [0.0, 1.0].
        x: Detected X position in the world/camera frame, in meters.
        y: Detected Y position in the world/camera frame, in meters.
        z: Detected Z position (altitude) in the world/camera frame,
            in meters.
        bbox_width: Width of the 2D bounding box in the source image,
            in pixels. Optional; defaults to 0.0 when unavailable
            (e.g. for point/range-sensor detections).
        bbox_height: Height of the 2D bounding box in the source
            image, in pixels. Optional; defaults to 0.0 when
            unavailable.
        timestamp: Time the detection was captured, in seconds
            (e.g. from a monotonic clock or ROS2 time).
    """

    detection_id: int = 0
    label: str = "unknown"
    confidence: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    bbox_width: float = 0.0
    bbox_height: float = 0.0
    timestamp: float = 0.0


@dataclass
class Track3D:
    """A persistent 3D object track built up from associated detections.

    Where `Detection` is a single frame's observation, `Track3D` is the
    temporally-associated identity that `ObjectManager` maintains across
    frames: a smoothed position, an estimated velocity, and bookkeeping
    about how long the track has existed and how recently it was seen.
    `MotionPredictor` and `EnvironmentModel` both consume `Track3D`
    objects rather than raw `Detection` objects.

    Attributes:
        track_id: Stable identifier for this track, persistent across
            frames for as long as the object continues to be
            associated with new detections.
        label: Class label of the tracked object (e.g. "person",
            "drone", "tree", "building").
        x: Current estimated X position, in meters.
        y: Current estimated Y position, in meters.
        z: Current estimated Z position (altitude), in meters.
        velocity_x: Estimated X velocity, in meters per second.
        velocity_y: Estimated Y velocity, in meters per second.
        velocity_z: Estimated Z velocity, in meters per second.
        confidence: Current confidence in this track's identity and
            position, in the range [0.0, 1.0].
        age: Number of frames/updates this track has existed for.
        last_seen: Timestamp, in seconds, of the most recent detection
            associated with this track.
    """

    track_id: int = 0
    label: str = "unknown"
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    velocity_z: float = 0.0
    confidence: float = 0.0
    age: int = 0
    last_seen: float = 0.0


@dataclass
class ObstacleEntry:
    """A single obstacle entry in the `EnvironmentModel`'s obstacle map.

    Obstacle entries are the world-model's representation of a hazard
    the UAV must plan and/or make decisions around. They may originate
    directly from a `Track3D` (a tracked dynamic object) or from static
    map/sensor data.

    Attributes:
        obstacle_id: Identifier for this obstacle. When derived from a
            track, this mirrors the originating `Track3D.track_id`.
        x: Obstacle X position, in meters.
        y: Obstacle Y position, in meters.
        z: Obstacle Z position (altitude), in meters.
        radius: Approximate obstacle radius, in meters, used for
            simple circular inflation/clearance checks during
            planning.
        is_dynamic: True if the obstacle is moving (derived from a
            tracked object with non-zero velocity), False if it is
            treated as static for planning purposes.
        risk_level: Coarse risk classification for this obstacle,
            e.g. "low", "medium", "high". Intended to let
            `DecisionEngine` prioritize which obstacles matter most.
        source_track_id: Identifier of the `Track3D` this obstacle
            entry was derived from, if any. `None` for obstacles that
            do not originate from a track (e.g. static map obstacles).
    """

    obstacle_id: int = 0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    radius: float = 0.0
    is_dynamic: bool = False
    risk_level: str = "low"
    source_track_id: "int | None" = None


@dataclass
class PlannerResult:
    """Output produced by `PlannerEngine` for a single planning cycle.

    Attributes:
        success: True if the planner found a usable path/route to the
            active waypoint or goal, False if planning failed (e.g. no
            feasible route given the current obstacle set).
        waypoints: Ordered list of `(x, y, z)` tuples describing the
            planned route, in meters, starting from (or near) the
            UAV's current position and ending at the planning target.
            Empty when `success` is False.
        total_cost: Aggregate cost of the planned route (e.g. path
            length in meters, optionally inflated by obstacle
            proximity). Meaningless when `success` is False.
        blocked_by: Identifier of the `ObstacleEntry` that caused
            planning to fail, if applicable. `None` when planning
            succeeded or failed for a reason unrelated to a specific
            obstacle.
    """

    success: bool = False
    waypoints: list = None
    total_cost: float = 0.0
    blocked_by: "int | None" = None

    def __post_init__(self) -> None:
        """Ensures `waypoints` defaults to an empty list per instance.

        A mutable default (`[]`) cannot be used directly as a
        dataclass field default, since that would share a single list
        across every `PlannerResult` instance. This assigns a fresh
        list to instances constructed without an explicit value.
        """
        if self.waypoints is None:
            self.waypoints = []


@dataclass
class DecisionResult:
    """Output produced by `DecisionEngine` for a single decision cycle.

    This is the Version 5 analogue of the string value passed to
    `NavigationCore.set_navigation_decision()`, but carries additional
    context (reason, and optional lateral offset / target waypoint)
    that a future integration pass can use when translating a decision
    into a velocity command.

    Attributes:
        decision: The chosen high-level maneuver. Uses the same
            vocabulary as `NavigationCore.navigation_decision`, i.e.
            one of "GO_TO_GOAL", "MOVE_LEFT", "MOVE_RIGHT", "ROTATE",
            or "STOP".
        reason: Short human-readable explanation of why this decision
            was chosen (e.g. "obstacle within safety radius",
            "path clear", "no active goal"). Intended for logging and
            debugging.
        target_waypoint: The `(x, y, z)` waypoint this decision is
            oriented towards, if applicable. `None` when the decision
            does not target a specific waypoint (e.g. "STOP").
        lateral_offset: Suggested lateral offset, in meters, to apply
            when the decision is "MOVE_LEFT" or "MOVE_RIGHT". Mirrors
            the fixed 0.3 m/s lateral velocities currently hard-coded
            in `NavigationCore.compute_navigation()`, expressed here as
            a distance rather than a velocity so `VelocityController`
            can decide how to realize it.
        confidence: Confidence in this decision, in the range
            [0.0, 1.0], based on the quality/coverage of the
            underlying `EnvironmentModel` and `PlannerResult`.
    """

    decision: str = "GO_TO_GOAL"
    reason: str = ""
    target_waypoint: "tuple | None" = None
    lateral_offset: float = 0.0
    confidence: float = 0.0


class MissionManager:
    """Manages mission-level state: an ordered sequence of goal legs.

    Where `NavigationCore` holds exactly one `Goal` at a time,
    `MissionManager` holds an ordered list of goal legs for a full
    mission and tracks which leg is currently active. It is a planning
    input for `WaypointManager` and `PlannerEngine`; it does not itself
    talk to `NavigationCore`.

    Attributes:
        legs: Ordered list of `(x, y, z)` tuples describing each goal
            leg of the mission, in meters.
        current_leg_index: Index into `legs` of the currently active
            leg. `-1` if no legs have been added yet.
    """

    def __init__(self) -> None:
        """Initializes an empty mission with no legs."""
        self.legs: list = []
        self.current_leg_index: int = -1

    def add_leg(self, x: float, y: float, z: float = 0.0) -> None:
        """Appends a new goal leg to the end of the mission.

        Args:
            x: Leg goal X coordinate, in meters.
            y: Leg goal Y coordinate, in meters.
            z: Leg goal Z coordinate (altitude), in meters.
        """
        self.legs.append((x, y, z))
        if self.current_leg_index == -1:
            self.current_leg_index = 0

    def current_leg(self) -> "tuple | None":
        """Returns the currently active leg, if any.

        Returns:
            The `(x, y, z)` tuple for the current leg, or `None` if
            there are no legs or the mission has been exhausted.
        """
        if 0 <= self.current_leg_index < len(self.legs):
            return self.legs[self.current_leg_index]
        return None

    def advance_leg(self) -> bool:
        """Advances to the next mission leg, if one exists.

        Returns:
            True if there was a next leg to advance to, False if the
            mission has already reached (or was at) its final leg.
        """
        if self.current_leg_index < len(self.legs) - 1:
            self.current_leg_index += 1
            return True
        return False

    def is_mission_complete(self) -> bool:
        """Checks whether every leg of the mission has been consumed.

        Returns:
            True if there are no legs, or the current leg index is
            already at the final leg (i.e. there is no further leg to
            advance to); False otherwise.
        """
        if len(self.legs) == 0:
            return True
        return self.current_leg_index >= len(self.legs) - 1

    def reset(self) -> None:
        """Clears all legs and returns the mission to its initial state."""
        self.legs = []
        self.current_leg_index = -1


class ObjectManager:
    """Associates per-frame `Detection` objects into persistent `Track3D` tracks.

    ObjectManager owns the set of currently active tracks. Each call to
    `update()` takes the latest batch of detections, associates them
    with existing tracks (by nearest-position matching within a
    configurable gate distance), updates matched tracks, creates new
    tracks for unmatched detections, and ages out tracks that have not
    been seen recently.

    Attributes:
        association_gate: Maximum distance, in meters, between a
            detection and an existing track's position for them to be
            considered a match.
        max_track_age_seconds: Maximum time, in seconds, a track may
            go without being matched to a new detection before it is
            dropped.
        tracks: Dictionary mapping `track_id` to `Track3D`, holding
            every currently active track.
    """

    def __init__(
        self,
        association_gate: float = 1.5,
        max_track_age_seconds: float = 2.0,
    ) -> None:
        """Initializes an empty object manager.

        Args:
            association_gate: Maximum matching distance, in meters,
                used when associating detections to existing tracks.
            max_track_age_seconds: Maximum time, in seconds, a track
                may remain unmatched before being dropped.
        """
        self.association_gate: float = association_gate
        self.max_track_age_seconds: float = max_track_age_seconds
        self.tracks: dict = {}
        self._next_track_id: int = 0

    def update(self, detections: list, timestamp: float) -> "list[Track3D]":
        """Associates new detections with tracks and updates track state.

        Args:
            detections: List of `Detection` objects observed in the
                current frame.
            timestamp: Current time, in seconds, used to stamp updated
                tracks and to age out stale ones.

        Returns:
            The list of `Track3D` objects that are active after this
            update (i.e. `list(self.tracks.values())`).
        """
        unmatched_detections = list(detections)

        for track in self.tracks.values():
            best_match = None
            best_distance = self.association_gate
            for detection in unmatched_detections:
                distance = math.sqrt(
                    (detection.x - track.x) ** 2
                    + (detection.y - track.y) ** 2
                    + (detection.z - track.z) ** 2
                )
                if distance <= best_distance:
                    best_distance = distance
                    best_match = detection

            if best_match is not None:
                dt = max(timestamp - track.last_seen, 1e-6)
                track.velocity_x = (best_match.x - track.x) / dt
                track.velocity_y = (best_match.y - track.y) / dt
                track.velocity_z = (best_match.z - track.z) / dt
                track.x = best_match.x
                track.y = best_match.y
                track.z = best_match.z
                track.confidence = best_match.confidence
                track.age += 1
                track.last_seen = timestamp
                unmatched_detections.remove(best_match)

        for detection in unmatched_detections:
            new_track = Track3D(
                track_id=self._next_track_id,
                label=detection.label,
                x=detection.x,
                y=detection.y,
                z=detection.z,
                velocity_x=0.0,
                velocity_y=0.0,
                velocity_z=0.0,
                confidence=detection.confidence,
                age=1,
                last_seen=timestamp,
            )
            self.tracks[new_track.track_id] = new_track
            self._next_track_id += 1

        stale_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if (timestamp - track.last_seen) > self.max_track_age_seconds
        ]
        for track_id in stale_ids:
            del self.tracks[track_id]

        return list(self.tracks.values())

    def get_track(self, track_id: int) -> "Track3D | None":
        """Looks up a single track by id.

        Args:
            track_id: Identifier of the track to look up.

        Returns:
            The matching `Track3D`, or `None` if no such track exists.
        """
        return self.tracks.get(track_id)

    def clear(self) -> None:
        """Removes all tracks and resets the track id counter."""
        self.tracks = {}
        self._next_track_id = 0


class MotionPredictor:
    """Predicts future positions of tracked objects using constant velocity.

    MotionPredictor is a stateless utility over `Track3D` data: given a
    track's current position/velocity estimate, it extrapolates a
    future position assuming the object continues along its current
    velocity vector. This is intentionally simple (no acceleration
    model, no maneuver classification) and is meant to feed
    `EnvironmentModel`'s obstacle inflation, not to be a full motion
    model.

    Attributes:
        default_horizon_seconds: Default prediction horizon, in
            seconds, used when `predict()` is called without an
            explicit horizon.
    """

    def __init__(self, default_horizon_seconds: float = 1.0) -> None:
        """Initializes the predictor with a default prediction horizon.

        Args:
            default_horizon_seconds: Default look-ahead time, in
                seconds, for constant-velocity prediction.
        """
        self.default_horizon_seconds: float = default_horizon_seconds

    def predict(self, track: Track3D, horizon_seconds: "float | None" = None) -> tuple:
        """Predicts a track's position after `horizon_seconds` seconds.

        Args:
            track: The `Track3D` whose future position is being
                predicted.
            horizon_seconds: How far ahead, in seconds, to predict.
                Defaults to `default_horizon_seconds` when omitted.

        Returns:
            An `(x, y, z)` tuple with the predicted position, in
            meters, assuming constant velocity.
        """
        horizon = (
            self.default_horizon_seconds
            if horizon_seconds is None
            else horizon_seconds
        )
        predicted_x = track.x + track.velocity_x * horizon
        predicted_y = track.y + track.velocity_y * horizon
        predicted_z = track.z + track.velocity_z * horizon
        return (predicted_x, predicted_y, predicted_z)

    def predict_all(
        self, tracks: "list[Track3D]", horizon_seconds: "float | None" = None
    ) -> dict:
        """Predicts future positions for a batch of tracks.

        Args:
            tracks: List of `Track3D` objects to predict.
            horizon_seconds: How far ahead, in seconds, to predict for
                every track. Defaults to `default_horizon_seconds`
                when omitted.

        Returns:
            A dictionary mapping `track_id` to the predicted
            `(x, y, z)` position tuple.
        """
        return {
            track.track_id: self.predict(track, horizon_seconds) for track in tracks
        }


class EnvironmentModel:
    """Maintains a world-model obstacle map derived from tracks and predictions.

    EnvironmentModel is the bridge between perception (`Track3D` /
    `MotionPredictor`) and planning/decision-making
    (`PlannerEngine` / `DecisionEngine`). It converts tracks into
    `ObstacleEntry` records, optionally inflated by predicted motion,
    and exposes simple clearance queries used during planning.

    Attributes:
        safety_margin: Additional radius, in meters, added on top of
            each obstacle's own radius when performing clearance
            checks.
        obstacles: Dictionary mapping `obstacle_id` to `ObstacleEntry`,
            holding the current obstacle map.
    """

    def __init__(self, safety_margin: float = 0.5) -> None:
        """Initializes an empty environment model.

        Args:
            safety_margin: Extra clearance radius, in meters, applied
                around every obstacle during clearance checks.
        """
        self.safety_margin: float = safety_margin
        self.obstacles: dict = {}

    def update_from_tracks(
        self,
        tracks: "list[Track3D]",
        predictor: "MotionPredictor | None" = None,
        default_radius: float = 0.5,
    ) -> None:
        """Rebuilds the obstacle map from the current set of tracks.

        Args:
            tracks: Currently active `Track3D` objects, typically the
                output of `ObjectManager.update()`.
            predictor: Optional `MotionPredictor` used to inflate each
                obstacle's effective position towards where the
                tracked object is predicted to be. When omitted, each
                obstacle is placed at the track's current position.
            default_radius: Radius, in meters, assigned to obstacles
                derived from tracks, since `Track3D` does not itself
                carry a size estimate.
        """
        self.obstacles = {}
        for track in tracks:
            if predictor is not None:
                predicted_x, predicted_y, predicted_z = predictor.predict(track)
            else:
                predicted_x, predicted_y, predicted_z = track.x, track.y, track.z

            is_dynamic = (
                abs(track.velocity_x) > 1e-6
                or abs(track.velocity_y) > 1e-6
                or abs(track.velocity_z) > 1e-6
            )
            risk_level = "high" if is_dynamic else "low"

            self.obstacles[track.track_id] = ObstacleEntry(
                obstacle_id=track.track_id,
                x=predicted_x,
                y=predicted_y,
                z=predicted_z,
                radius=default_radius,
                is_dynamic=is_dynamic,
                risk_level=risk_level,
                source_track_id=track.track_id,
            )

    def add_static_obstacle(
        self, obstacle_id: int, x: float, y: float, z: float, radius: float
    ) -> None:
        """Adds (or replaces) a static obstacle not derived from a track.

        Args:
            obstacle_id: Identifier for the obstacle.
            x: Obstacle X position, in meters.
            y: Obstacle Y position, in meters.
            z: Obstacle Z position, in meters.
            radius: Obstacle radius, in meters.
        """
        self.obstacles[obstacle_id] = ObstacleEntry(
            obstacle_id=obstacle_id,
            x=x,
            y=y,
            z=z,
            radius=radius,
            is_dynamic=False,
            risk_level="low",
            source_track_id=None,
        )

    def nearest_obstacle(
        self, x: float, y: float, z: float = 0.0
    ) -> "ObstacleEntry | None":
        """Finds the obstacle closest to a given point.

        Args:
            x: Query X position, in meters.
            y: Query Y position, in meters.
            z: Query Z position, in meters.

        Returns:
            The closest `ObstacleEntry`, or `None` if the obstacle map
            is empty.
        """
        if not self.obstacles:
            return None
        return min(
            self.obstacles.values(),
            key=lambda obstacle: math.sqrt(
                (obstacle.x - x) ** 2 + (obstacle.y - y) ** 2 + (obstacle.z - z) ** 2
            ),
        )

    def is_point_clear(self, x: float, y: float, z: float = 0.0) -> bool:
        """Checks whether a point is clear of every known obstacle.

        A point is considered clear if its distance to every
        obstacle's center exceeds that obstacle's radius plus
        `safety_margin`.

        Args:
            x: Query X position, in meters.
            y: Query Y position, in meters.
            z: Query Z position, in meters.

        Returns:
            True if the point clears every obstacle, False if it
            falls within any obstacle's inflated radius.
        """
        for obstacle in self.obstacles.values():
            distance = math.sqrt(
                (obstacle.x - x) ** 2 + (obstacle.y - y) ** 2 + (obstacle.z - z) ** 2
            )
            if distance <= (obstacle.radius + self.safety_margin):
                return False
        return True

    def clear(self) -> None:
        """Removes every obstacle from the model."""
        self.obstacles = {}


class WaypointManager:
    """Manages an ordered queue of waypoints for the planner to consume.

    WaypointManager sits between `MissionManager` (which describes a
    mission as a sequence of goal legs) and `PlannerEngine` (which
    needs a concrete queue of waypoints to route through). It can be
    seeded directly, or populated one leg at a time from a
    `MissionManager`.

    Attributes:
        waypoints: Ordered list of `(x, y, z)` tuples, in meters,
            still to be visited.
        visited: Ordered list of `(x, y, z)` tuples that have already
            been popped off `waypoints` via `advance()`.
    """

    def __init__(self) -> None:
        """Initializes an empty waypoint queue."""
        self.waypoints: list = []
        self.visited: list = []

    def set_waypoints(self, waypoints: list) -> None:
        """Replaces the waypoint queue wholesale.

        Args:
            waypoints: New ordered list of `(x, y, z)` tuples, in
                meters.
        """
        self.waypoints = list(waypoints)
        self.visited = []

    def load_from_mission(self, mission_manager: "MissionManager") -> None:
        """Populates the waypoint queue from a `MissionManager`'s legs.

        Args:
            mission_manager: Source of the ordered `(x, y, z)` leg
                tuples to load as waypoints.
        """
        self.waypoints = list(mission_manager.legs)
        self.visited = []

    def current_waypoint(self) -> "tuple | None":
        """Returns the next waypoint to fly towards, without removing it.

        Returns:
            The `(x, y, z)` tuple at the front of the queue, or `None`
            if the queue is empty.
        """
        if self.waypoints:
            return self.waypoints[0]
        return None

    def advance(self) -> "tuple | None":
        """Pops the current waypoint off the queue and marks it visited.

        Returns:
            The waypoint that was just popped, or `None` if the queue
            was already empty.
        """
        if not self.waypoints:
            return None
        waypoint = self.waypoints.pop(0)
        self.visited.append(waypoint)
        return waypoint

    def is_empty(self) -> bool:
        """Checks whether there are any waypoints left to visit.

        Returns:
            True if the waypoint queue is empty, False otherwise.
        """
        return len(self.waypoints) == 0

    def reset(self) -> None:
        """Clears both the pending and visited waypoint lists."""
        self.waypoints = []
        self.visited = []


class PlannerEngine:
    """Computes a route to the active waypoint given the current obstacle map.

    PlannerEngine performs simple, geometry-based route planning: it
    checks whether a direct straight-line path from the current
    position to the target waypoint is clear according to the
    `EnvironmentModel`, and if not, attempts a small set of lateral
    detour offsets before reporting failure. This mirrors the
    lightweight, proportional-control spirit of `NavigationCore`
    rather than implementing a full grid-search planner such as A*.

    Attributes:
        detour_offsets: Candidate lateral offsets, in meters, tried
            (perpendicular to the direct path) when the direct path is
            blocked.
        sample_step: Distance, in meters, between points sampled along
            a candidate path when checking it for clearance.
    """

    def __init__(
        self,
        detour_offsets: "list[float] | None" = None,
        sample_step: float = 0.5,
    ) -> None:
        """Initializes the planner with detour and sampling parameters.

        Args:
            detour_offsets: Candidate lateral offsets, in meters, to
                try when the direct path to a waypoint is blocked.
                Defaults to `[1.0, -1.0, 2.0, -2.0]` when omitted.
            sample_step: Distance, in meters, between clearance-check
                samples along a candidate path.
        """
        self.detour_offsets: list = (
            [1.0, -1.0, 2.0, -2.0] if detour_offsets is None else list(detour_offsets)
        )
        self.sample_step: float = sample_step

    def _sample_path_clear(
        self,
        start: tuple,
        end: tuple,
        environment_model: "EnvironmentModel",
    ) -> bool:
        """Checks whether every sampled point along a straight path is clear.

        Args:
            start: `(x, y, z)` start point, in meters.
            end: `(x, y, z)` end point, in meters.
            environment_model: Obstacle map to check clearance against.

        Returns:
            True if every sampled point between `start` and `end` is
            clear of obstacles, False otherwise.
        """
        distance = math.sqrt(
            (end[0] - start[0]) ** 2
            + (end[1] - start[1]) ** 2
            + (end[2] - start[2]) ** 2
        )
        if distance <= 1e-9:
            return environment_model.is_point_clear(*start)

        steps = max(1, int(distance / self.sample_step))
        for step in range(steps + 1):
            t = step / steps
            point = (
                start[0] + (end[0] - start[0]) * t,
                start[1] + (end[1] - start[1]) * t,
                start[2] + (end[2] - start[2]) * t,
            )
            if not environment_model.is_point_clear(*point):
                return False
        return True

    def plan(
        self,
        current_position: tuple,
        target_waypoint: tuple,
        environment_model: "EnvironmentModel",
    ) -> PlannerResult:
        """Plans a route from the current position to a target waypoint.

        Attempts the direct straight-line path first. If that path is
        blocked, tries a small set of laterally-offset two-segment
        detours (current position -> offset midpoint -> target) drawn
        from `detour_offsets`, in order, and returns the first one
        found clear.

        Args:
            current_position: `(x, y, z)` current UAV position, in
                meters.
            target_waypoint: `(x, y, z)` target waypoint, in meters.
            environment_model: Obstacle map to plan against.

        Returns:
            A `PlannerResult` describing the planned route. `success`
            is False, with `blocked_by` set to the id of the nearest
            obstacle to the target, if no clear route could be found.
        """
        if self._sample_path_clear(
            current_position, target_waypoint, environment_model
        ):
            distance = math.sqrt(
                (target_waypoint[0] - current_position[0]) ** 2
                + (target_waypoint[1] - current_position[1]) ** 2
                + (target_waypoint[2] - current_position[2]) ** 2
            )
            return PlannerResult(
                success=True,
                waypoints=[current_position, target_waypoint],
                total_cost=distance,
                blocked_by=None,
            )

        dx = target_waypoint[0] - current_position[0]
        dy = target_waypoint[1] - current_position[1]
        heading = math.atan2(dy, dx)
        perpendicular = heading + (math.pi / 2.0)

        for offset in self.detour_offsets:
            midpoint = (
                (current_position[0] + target_waypoint[0]) / 2.0
                + math.cos(perpendicular) * offset,
                (current_position[1] + target_waypoint[1]) / 2.0
                + math.sin(perpendicular) * offset,
                (current_position[2] + target_waypoint[2]) / 2.0,
            )
            first_leg_clear = self._sample_path_clear(
                current_position, midpoint, environment_model
            )
            second_leg_clear = self._sample_path_clear(
                midpoint, target_waypoint, environment_model
            )
            if first_leg_clear and second_leg_clear:
                leg_one_distance = math.sqrt(
                    (midpoint[0] - current_position[0]) ** 2
                    + (midpoint[1] - current_position[1]) ** 2
                    + (midpoint[2] - current_position[2]) ** 2
                )
                leg_two_distance = math.sqrt(
                    (target_waypoint[0] - midpoint[0]) ** 2
                    + (target_waypoint[1] - midpoint[1]) ** 2
                    + (target_waypoint[2] - midpoint[2]) ** 2
                )
                return PlannerResult(
                    success=True,
                    waypoints=[current_position, midpoint, target_waypoint],
                    total_cost=leg_one_distance + leg_two_distance,
                    blocked_by=None,
                )

        blocking_obstacle = environment_model.nearest_obstacle(*target_waypoint)
        return PlannerResult(
            success=False,
            waypoints=[],
            total_cost=0.0,
            blocked_by=(
                blocking_obstacle.obstacle_id if blocking_obstacle is not None else None
            ),
        )


class DecisionEngine:
    """Turns environment and planner state into a single high-level decision.

    DecisionEngine is the Version 5 analogue of the informal "Decision
    Engine" referenced in `NavigationCore.set_navigation_decision()`'s
    docstring. It inspects the nearest obstacle in the
    `EnvironmentModel` together with the latest `PlannerResult` and
    produces a `DecisionResult` using the same decision vocabulary
    ("GO_TO_GOAL", "MOVE_LEFT", "MOVE_RIGHT", "ROTATE", "STOP") that
    `NavigationCore` already understands, without calling into
    `NavigationCore` itself.

    Attributes:
        stop_distance: Distance, in meters, inside of which an
            obstacle triggers a "STOP" decision.
        avoid_distance: Distance, in meters, inside of which an
            obstacle (but outside `stop_distance`) triggers a
            "MOVE_LEFT"/"MOVE_RIGHT" avoidance decision.
    """

    def __init__(self, stop_distance: float = 0.75, avoid_distance: float = 2.0) -> None:
        """Initializes the decision engine's distance thresholds.

        Args:
            stop_distance: Obstacle distance, in meters, at or inside
                of which the engine decides "STOP".
            avoid_distance: Obstacle distance, in meters, at or inside
                of which (but outside `stop_distance`) the engine
                decides to laterally avoid the obstacle.
        """
        self.stop_distance: float = stop_distance
        self.avoid_distance: float = avoid_distance

    def decide(
        self,
        current_position: tuple,
        environment_model: "EnvironmentModel",
        planner_result: "PlannerResult | None" = None,
    ) -> DecisionResult:
        """Produces a single high-level decision for the current cycle.

        Args:
            current_position: `(x, y, z)` current UAV position, in
                meters.
            environment_model: Obstacle map to evaluate proximity
                hazards against.
            planner_result: Most recent `PlannerResult`, if available.
                When provided and `success` is True, its second
                waypoint (or last, if only two) is used as the decision's
                `target_waypoint`.

        Returns:
            A `DecisionResult` describing the chosen maneuver, the
            reason for it, and any relevant target/offset context.
        """
        nearest = environment_model.nearest_obstacle(*current_position)

        target_waypoint = None
        if planner_result is not None and planner_result.success and planner_result.waypoints:
            target_waypoint = planner_result.waypoints[-1]

        if nearest is None:
            return DecisionResult(
                decision="GO_TO_GOAL",
                reason="no obstacles in environment model",
                target_waypoint=target_waypoint,
                lateral_offset=0.0,
                confidence=1.0,
            )

        distance = math.sqrt(
            (nearest.x - current_position[0]) ** 2
            + (nearest.y - current_position[1]) ** 2
            + (nearest.z - current_position[2]) ** 2
        )

        if distance <= self.stop_distance:
            return DecisionResult(
                decision="STOP",
                reason=f"obstacle {nearest.obstacle_id} within stop distance",
                target_waypoint=target_waypoint,
                lateral_offset=0.0,
                confidence=1.0,
            )

        if distance <= self.avoid_distance:
            dx = current_position[0] - nearest.x
            dy = current_position[1] - nearest.y
            cross = dx * math.cos(0.0) - dy * math.sin(0.0)
            decision = "MOVE_LEFT" if cross >= 0 else "MOVE_RIGHT"
            return DecisionResult(
                decision=decision,
                reason=f"obstacle {nearest.obstacle_id} within avoid distance",
                target_waypoint=target_waypoint,
                lateral_offset=0.3,
                confidence=0.75,
            )

        if planner_result is not None and not planner_result.success:
            return DecisionResult(
                decision="ROTATE",
                reason="planner could not find a clear route",
                target_waypoint=target_waypoint,
                lateral_offset=0.0,
                confidence=0.5,
            )

        return DecisionResult(
            decision="GO_TO_GOAL",
            reason="nearest obstacle outside avoid distance",
            target_waypoint=target_waypoint,
            lateral_offset=0.0,
            confidence=0.9,
        )


class VelocityController:
    """Converts a `DecisionResult` into velocity-command-shaped output.

    VelocityController is the Version 5 counterpart to
    `NavigationCore._generate_velocity_command()` /
    `NavigationCore.compute_navigation()`: given a `DecisionResult` and
    the same proportional-control parameters NavigationCore uses, it
    computes linear/angular velocities. It is deliberately independent
    of `NavigationCore` for now (per the current integration scope) and
    returns a plain dictionary rather than a `NavigationCommand`, so it
    can be exercised and tested without constructing a
    `NavigationCore` instance.

    Attributes:
        max_linear_speed: Maximum linear speed, in meters per second.
        max_angular_speed: Maximum angular speed, in radians per
            second.
        kp_linear: Proportional gain applied to distance-to-target.
        kp_angular: Proportional gain applied to heading error.
    """

    def __init__(
        self,
        max_linear_speed: float = 1.0,
        max_angular_speed: float = 1.0,
        kp_linear: float = 0.5,
        kp_angular: float = 1.0,
    ) -> None:
        """Initializes the controller with the same gains/limits NavigationCore uses.

        Args:
            max_linear_speed: Maximum linear speed, in meters per
                second. Must be positive.
            max_angular_speed: Maximum angular speed, in radians per
                second. Must be positive.
            kp_linear: Proportional gain for linear speed
                (speed = kp_linear * distance).
            kp_angular: Proportional gain for angular speed
                (speed = kp_angular * heading_error).
        """
        self.max_linear_speed: float = max_linear_speed
        self.max_angular_speed: float = max_angular_speed
        self.kp_linear: float = kp_linear
        self.kp_angular: float = kp_angular

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalizes an angle to the range [-pi, pi].

        Args:
            angle: Angle in radians to normalize.

        Returns:
            The equivalent angle wrapped into [-pi, pi].
        """
        return math.atan2(math.sin(angle), math.cos(angle))

    def generate_command(
        self,
        current_position: tuple,
        current_yaw: float,
        decision_result: "DecisionResult",
    ) -> dict:
        """Generates a velocity command dictionary from a `DecisionResult`.

        Args:
            current_position: `(x, y, z)` current UAV position, in
                meters.
            current_yaw: Current UAV heading, in radians.
            decision_result: The `DecisionResult` to realize as a
                velocity command.

        Returns:
            A dictionary with keys `linear_x`, `linear_y`, `linear_z`,
            and `angular_z` (all floats, in the same units/semantics
            as `NavigationCommand`), shaped so it can be used to
            construct a `NavigationCommand` in a future integration
            pass.
        """
        if decision_result.decision == "STOP":
            return {
                "linear_x": 0.0,
                "linear_y": 0.0,
                "linear_z": 0.0,
                "angular_z": 0.0,
            }

        if decision_result.target_waypoint is None:
            return {
                "linear_x": 0.0,
                "linear_y": (
                    decision_result.lateral_offset
                    if decision_result.decision == "MOVE_LEFT"
                    else (
                        -decision_result.lateral_offset
                        if decision_result.decision == "MOVE_RIGHT"
                        else 0.0
                    )
                ),
                "linear_z": 0.0,
                "angular_z": (
                    self.max_angular_speed
                    if decision_result.decision == "ROTATE"
                    else 0.0
                ),
            }

        dx = decision_result.target_waypoint[0] - current_position[0]
        dy = decision_result.target_waypoint[1] - current_position[1]
        distance = math.hypot(dx, dy)
        heading_to_target = math.atan2(dy, dx)
        heading_error = self._normalize_angle(heading_to_target - current_yaw)

        linear_x = max(
            0.0, min(self.kp_linear * distance, self.max_linear_speed)
        )
        angular_z = max(
            -self.max_angular_speed,
            min(self.kp_angular * heading_error, self.max_angular_speed),
        )

        linear_y = 0.0
        if decision_result.decision == "MOVE_LEFT":
            linear_y = decision_result.lateral_offset
        elif decision_result.decision == "MOVE_RIGHT":
            linear_y = -decision_result.lateral_offset
        elif decision_result.decision == "ROTATE":
            linear_x = 0.0
            angular_z = self.max_angular_speed

        return {
            "linear_x": linear_x,
            "linear_y": linear_y,
            "linear_z": 0.0,
            "angular_z": angular_z,
        }


class NavigationCore:
    """Computes direct-to-goal velocity commands for a UAV.

    NavigationCore holds the current mission goal and a small set of
    tunable flight parameters. Given the UAV's current pose, it
    computes the distance and heading to the goal and derives a
    proportional velocity command that drives the UAV straight
    towards the goal. When the UAV is within the configured goal
    tolerance, the class reports that the goal has been reached and
    returns zero velocities so the UAV hovers in place.

    Navigation currently operates in the X/Y plane only. Z is stored
    on both `Pose` and `Goal` so altitude control can be added in a
    later phase without reshaping the API.

    Attributes:
        max_linear_speed: Maximum linear speed the UAV is allowed to
            command, in meters per second.
        max_angular_speed: Maximum angular speed the UAV is allowed to
            command, in radians per second.
        goal_tolerance: Distance, in meters, within which the goal is
            considered reached.
        kp_linear: Proportional gain applied to distance-to-goal when
            computing linear speed.
        kp_angular: Proportional gain applied to heading error when
            computing angular speed.
        goal: Current mission goal.
        pose: UAV's current pose.
        state: Current `MissionState`.
    """

    def __init__(
        self,
        max_linear_speed: float = 1.0,
        max_angular_speed: float = 1.0,
        goal_tolerance: float = 0.5,
        kp_linear: float = 0.5,
        kp_angular: float = 1.0,
    ) -> None:
        """Initializes the navigation core with default flight limits.

        Args:
            max_linear_speed: Maximum linear speed in meters per
                second. Must be positive.
            max_angular_speed: Maximum angular speed in radians per
                second. Must be positive.
            goal_tolerance: Radius in meters around the goal within
                which the UAV is considered to have arrived. Must be
                non-negative.
            kp_linear: Proportional gain for the linear speed
                controller (speed = kp_linear * distance).
            kp_angular: Proportional gain for the angular speed
                controller (speed = kp_angular * heading_error).
        """
        self.max_linear_speed: float = max_linear_speed
        self.max_angular_speed: float = max_angular_speed
        self.goal_tolerance: float = goal_tolerance
        self.kp_linear: float = kp_linear
        self.kp_angular: float = kp_angular

        self.goal: Goal = Goal()
        self.pose: Pose = Pose()
        self.state: MissionState = MissionState.IDLE

        # Decision Engine integration point. Set externally via
        # set_navigation_decision() (e.g. from opencv_navigation_v4.py's
        # Decision Engine). compute_navigation() consults this before
        # running its normal goal-seeking logic. Defaults to GO_TO_GOAL so
        # behaviour is identical to before this was added until something
        # explicitly calls set_navigation_decision().
        self.navigation_decision: str = "GO_TO_GOAL"

    # ------------------------------------------------------------------
    # Mission setup
    # ------------------------------------------------------------------

    def set_goal(self, x: float, y: float, z: float = 0.0) -> None:
        """Sets (or replaces) the current mission goal position.

        Args:
            x: Goal X coordinate, in meters.
            y: Goal Y coordinate, in meters.
            z: Goal Z coordinate (altitude), in meters. Stored for
                future altitude control; not used for Phase 1 motion.
        """
        self.goal = Goal(x=x, y=y, z=z)
        self.state = MissionState.GO_TO_GOAL

    def update_pose(self, x: float, y: float, yaw: float, z: float = 0.0) -> None:
        """Updates the UAV's current pose.

        Args:
            x: Current X coordinate of the UAV, in meters.
            y: Current Y coordinate of the UAV, in meters.
            yaw: Current yaw (heading) of the UAV, in radians. Any
                value is accepted since angle differences are
                normalized internally.
            z: Current Z coordinate (altitude) of the UAV, in meters.
                Stored for future altitude control; not used for
                Phase 1 motion.
        """
        self.pose = Pose(x=x, y=y, z=z, yaw=yaw)

    def set_navigation_decision(self, decision: str) -> None:
        """Updates the navigation decision consumed by compute_navigation().

        This is the sole integration point for an external Decision Engine
        (e.g. one built on top of Phase 4 tracking/prediction data in
        opencv_navigation_v4.py). It performs no validation and no other
        logic — it just stores the value. An unrecognized string behaves
        the same as "GO_TO_GOAL" in compute_navigation().

        Args:
            decision: One of "GO_TO_GOAL", "MOVE_LEFT", "MOVE_RIGHT",
                "ROTATE", or "STOP".
        """
        self.navigation_decision = decision

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _compute_geometry(self) -> _NavigationGeometry:
        """Computes all goal-relative geometry in a single pass.

        This is the single source of truth for dx, dy, distance,
        heading, and heading error so no navigation math is repeated
        elsewhere in the class.

        Returns:
            A `_NavigationGeometry` snapshot for the current pose and
            goal.
        """
        dx = self.goal.x - self.pose.x
        dy = self.goal.y - self.pose.y
        distance = math.hypot(dx, dy)
        heading_to_goal = math.atan2(dy, dx)
        heading_error = self._normalize_angle(heading_to_goal - self.pose.yaw)
        return _NavigationGeometry(
            dx=dx,
            dy=dy,
            distance=distance,
            heading_to_goal=heading_to_goal,
            heading_error=heading_error,
        )

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalizes an angle to the range [-pi, pi].

        Args:
            angle: Angle in radians to normalize.

        Returns:
            The equivalent angle wrapped into [-pi, pi].
        """
        return math.atan2(math.sin(angle), math.cos(angle))

    def distance_to_goal(self) -> float:
        """Computes the straight-line distance from the UAV to the goal.

        Returns:
            The Euclidean (X/Y) distance to the goal, in meters.
        """
        return self._compute_geometry().distance

    def goal_reached(self) -> bool:
        """Checks whether the UAV is within tolerance of the goal.

        Returns:
            True if the current distance to the goal is less than or
            equal to `goal_tolerance`, False otherwise.
        """
        return self.distance_to_goal() <= self.goal_tolerance

    # ------------------------------------------------------------------
    # Velocity generation
    # ------------------------------------------------------------------

    def _generate_velocity_command(
        self, distance: float, heading_error: float
    ) -> tuple:
        """Generates a clamped, proportional velocity command.

        Args:
            distance: Current distance to the goal, in meters.
            heading_error: Normalized heading error, in radians.

        Returns:
            A tuple (linear_x, angular_z) clamped to
            `max_linear_speed` and `max_angular_speed` respectively.
        """
        linear_x = self.kp_linear * distance
        linear_x = max(0.0, min(linear_x, self.max_linear_speed))

        angular_z = self.kp_angular * heading_error
        angular_z = max(
            -self.max_angular_speed,
            min(angular_z, self.max_angular_speed),
        )
        return linear_x, angular_z

    def compute_navigation(self) -> NavigationCommand:
        """Computes the velocity command to fly towards the goal.

        Uses the UAV's current pose (set via `update_pose`) and the
        stored goal (set via `set_goal`) to compute distance, heading,
        and heading error, then derives a proportional velocity
        command:

            linear_x  = clamp(kp_linear  * distance,      max_linear_speed)
            angular_z = clamp(kp_angular * heading_error,  max_angular_speed)

        If the UAV is within `goal_tolerance` of the goal, the mission
        state becomes `GOAL_REACHED` and zero velocities are returned
        so the UAV hovers. Otherwise the mission state is
        `GO_TO_GOAL`.

        Returns:
            A `NavigationCommand` describing the velocity to apply and
            the current navigation status.
        """
        geometry = self._compute_geometry()
        goal_reached_now = geometry.distance <= self.goal_tolerance

        # ── Decision Engine override (checked before normal goal-seeking) ──
        # Geometry, angle normalization, distance/goal checking, and the
        # proportional controller below are unchanged; this just decides
        # which velocities to wrap into the returned NavigationCommand.
        if self.navigation_decision == "STOP":
            return NavigationCommand(
                linear_x=0.0,
                linear_y=0.0,
                linear_z=0.0,
                angular_z=0.0,
                goal_reached=goal_reached_now,
                distance_to_goal=geometry.distance,
            )

        if self.navigation_decision == "MOVE_LEFT":
            linear_x, angular_z = self._generate_velocity_command(
                geometry.distance, geometry.heading_error
            )
            return NavigationCommand(
                linear_x=linear_x,
                linear_y=0.3,
                linear_z=0.0,
                angular_z=angular_z,
                goal_reached=goal_reached_now,
                distance_to_goal=geometry.distance,
            )

        if self.navigation_decision == "MOVE_RIGHT":
            linear_x, angular_z = self._generate_velocity_command(
                geometry.distance, geometry.heading_error
            )
            return NavigationCommand(
                linear_x=linear_x,
                linear_y=-0.3,
                linear_z=0.0,
                angular_z=angular_z,
                goal_reached=goal_reached_now,
                distance_to_goal=geometry.distance,
            )

        if self.navigation_decision == "ROTATE":
            return NavigationCommand(
                linear_x=0.0,
                linear_y=0.0,
                linear_z=0.0,
                angular_z=self.max_angular_speed,
                goal_reached=goal_reached_now,
                distance_to_goal=geometry.distance,
            )

        # ── GO_TO_GOAL (default) — existing behaviour, unchanged ────────────
        if geometry.distance <= self.goal_tolerance:
            self.state = MissionState.GOAL_REACHED
            return NavigationCommand(
                linear_x=0.0,
                linear_y=0.0,
                linear_z=0.0,
                angular_z=0.0,
                goal_reached=True,
                distance_to_goal=geometry.distance,
            )

        self.state = MissionState.GO_TO_GOAL
        linear_x, angular_z = self._generate_velocity_command(
            geometry.distance, geometry.heading_error
        )

        return NavigationCommand(
            linear_x=linear_x,
            linear_y=0.0,
            linear_z=0.0,
            angular_z=angular_z,
            goal_reached=False,
            distance_to_goal=geometry.distance,
        )

    # ------------------------------------------------------------------
    # Mission management utilities
    # ------------------------------------------------------------------

    def stop(self) -> NavigationCommand:
        """Returns a zero-velocity command without altering the goal.

        Useful for emergency stops or transitions between mission
        phases where the UAV should hold position but the current
        goal and pose should be preserved.

        Returns:
            A `NavigationCommand` with all velocities set to zero and
            `distance_to_goal` reflecting the current, unchanged goal.
        """
        distance = self.distance_to_goal()
        return NavigationCommand(
            linear_x=0.0,
            linear_y=0.0,
            linear_z=0.0,
            angular_z=0.0,
            goal_reached=self.goal_reached(),
            distance_to_goal=distance,
        )

    def clear_goal(self) -> None:
        """Clears the current goal and returns the mission to IDLE.

        The UAV's pose is left untouched; only the goal and mission
        state are reset.
        """
        self.goal = Goal()
        self.state = MissionState.IDLE

    def reset(self) -> None:
        """Resets the navigation core to its initial, idle state.

        Clears both the goal and the pose and returns the mission
        state to IDLE. Flight parameters (speed limits, tolerance,
        gains) are left unchanged.
        """
        self.goal = Goal()
        self.pose = Pose()
        self.state = MissionState.IDLE