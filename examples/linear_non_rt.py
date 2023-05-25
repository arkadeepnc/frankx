from argparse import ArgumentParser

from frankx import Affine, LinearRelativeMotion, Robot, RealtimeConfig
import numpy as np


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--host', default='172.16.0.2', help='FCI IP of the robot')
    args = parser.parse_args()

    # Connect to the robot
    robot = Robot(args.host, realtime_config=RealtimeConfig.Ignore)
    robot.set_default_behavior()
    robot.recover_from_errors()

    # Reduce the acceleration and velocity dynamic
    # robot.set_dynamic_rel(0.05)
    robot.velocity_rel = 0.01

    # Define and move forwards
    way = Affine(0.1, 0.0, 0.0, 0, 0, 0)
    motion_forward = LinearRelativeMotion(way)
    robot.move(motion_forward)

    # And move backwards using the inverse motion
    # motion_backward = LinearRelativeMotion(way.inverse())
    # robot.move(motion_backward)
