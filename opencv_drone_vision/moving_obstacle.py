#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import subprocess
import math

class MovingObstacle(Node):

    def __init__(self):
        super().__init__('moving_obstacle')

        self.t = 0.0

        self.timer = self.create_timer(
            0.05,
            self.update_obstacles
        )

        self.get_logger().info(
            'Moving obstacles started'
        )

    def set_pose(self, name, x, y, z):

        cmd = f'''
gz service -s /world/drone_arena/set_pose \
--reqtype gz.msgs.Pose \
--reptype gz.msgs.Boolean \
--timeout 1000 \
--req 'name: "{name}" position {{ x:{x} y:{y} z:{z} }}'
'''

        subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    def update_obstacles(self):

        self.t += 0.15

        # chair_1 – patrols the entry corridor (Zone A), small radius
        chair1_x = 1.5 + 0.5 * math.cos(0.4 * self.t)
        chair1_y = 0.0 + 0.6 * math.sin(0.4 * self.t)

        # chair_2 – sweeps the upper branch (Zone B), crosses the corridor
        chair2_x = 3.2 + 0.7 * math.cos(-0.35 * self.t)
        chair2_y = 0.9 + 0.6 * math.sin(-0.35 * self.t)

        # chair_3 – slow pendulum across the central squeeze (Zone D)
        chair3_x = 5.5 + 0.5 * math.sin(0.3 * self.t)
        chair3_y = -0.5 + 0.6 * math.sin(0.5 * self.t)

        # chair_4 – patrols the final gate approach (Zone E)
        chair4_x = 7.2 + 0.5 * math.sin(0.4 * self.t)
        chair4_y = 0.0 + 0.7 * math.sin(0.55 * self.t)

        self.set_pose("chair_1", chair1_x, chair1_y, 0.0)
        self.set_pose("chair_2", chair2_x, chair2_y, 0.0)
        self.set_pose("chair_3", chair3_x, chair3_y, 0.0)
        self.set_pose("chair_4", chair4_x, chair4_y, 0.0)


def main(args=None):

    rclpy.init(args=args)

    node = MovingObstacle()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()