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