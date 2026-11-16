#!/usr/bin/env python3
"""
Original version created by Lukas Nawka. This version was adapted and is maintained by Niklas Leukroth.
"""

import sys

import rclpy
import math
from rclpy.node import Node
from std_msgs.msg import Bool
from lasr_msgs.msg import XArmCommand
from xarm_msgs.srv import MoveCartesian, Call, SetInt16, SetInt16ById
from xarm_msgs.msg import RobotMsg

class xArmMover(Node):

    def __init__(self):
        super().__init__("xarm_mover_node")
        self.client_move = self.create_client(MoveCartesian, "/xarm/set_position")
        self.client_state = self.create_client(SetInt16, "/xarm/set_state")
        self.client_mode = self.create_client(SetInt16, "/xarm/set_mode")
        self.client_clean_error = self.create_client(Call, "/xarm/clean_error")
        self.client_motion_enable = self.create_client(SetInt16ById, "/xarm/motion_enable")
        self.commandSub = self.create_subscription(XArmCommand, "/xarm/command", self.callback_model, 10)
        self.xarmSub = self.create_subscription(RobotMsg, "/xarm/robot_states", self.xarm_state_cb, 10)
        self.xarm_reset_pub = self.create_publisher(Bool, '/xarm/reset', 1)
        self.client_futures = []

    def reset_arm(self):
        error_req = Call.Request()
        self.client_clean_error.call_async(error_req)

        motion_request = SetInt16ById.Request()
        motion_request.id = 8
        motion_request.data = 1
        self.client_motion_enable.call_async(motion_request)

        mode_req = SetInt16.Request()
        mode_req.data = 7
        self.client_mode.call_async(mode_req)

        state_req = SetInt16.Request()
        state_req.data = 0
        self.client_state.call_async(state_req)


    def xarm_state_cb(self, msg):
        if msg.mode == 0:
            out = Bool()
            out.data = True
            self.xarm_reset_pub.publish(out)
            self.reset_arm()
            out.data = False
            self.xarm_reset_pub.publish(out)

    def callback_model(self, msg):
        state_req = SetInt16.Request()
        state_req.data = 0
        self.client_state.call_async(state_req)

        if msg.pos_x == 500.0 and msg.pos_z == 0.0:
            print("lost detection")
            return

        self.req = MoveCartesian.Request()
        self.req.pose = [msg.pos_x, msg.pos_y, msg.pos_z if msg.pos_z > 160.0 else 160.0, msg.rot_x, msg.rot_y, msg.rot_z]

        self.req.speed = msg.speed
        self.req.acc = msg.acc
        self.req.mvtime = msg.mvtime
        
        self.client_futures.append(self.client_move.call_async(self.req))
     

    def spin(self):
        while rclpy.ok():
            rclpy.spin_once(self)
            incomplete_futures = []
            for f in self.client_futures:
                if not f.done():
                    incomplete_futures.append(f)
            self.client_futures = incomplete_futures


def main(args=None):
    rclpy.init(args=args)
    node = xArmMover()
    node.spin()

if __name__ == "__main__":
    sys.exit(main())    
