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

        chair1_x = 2.5 + 1.2 * math.cos(0.6 * self.t)
        chair1_y = 0.0 + 1.2 * math.sin(0.6 * self.t)

        chair2_x = 5.0 + 1.0 * math.cos(-0.5 * self.t)
        chair2_y = 1.5 + 1.0 * math.sin(-0.5 * self.t)

        chair3_x = 4.0 + 1.2 * math.sin(0.7 * self.t)
        chair3_y = -1.0

        chair4_x = 6.5 + 1.4 * math.sin(0.45 * self.t)
        chair4_y = 0.5 + 0.7 * math.sin(0.9 * self.t)

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