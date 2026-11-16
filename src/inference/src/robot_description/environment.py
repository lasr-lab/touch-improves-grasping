import numpy as np
import collections
import math

from .robot_utils import XArm, Tilburg, IntelRealSense, Digit360

from std_msgs.msg import Float64MultiArray

from lasr_msgs.msg import XArmCommand

import IPython
e = IPython.embed

class RealEnv:

    def __init__(self, node):
        self.node = node
        self.robot_arm_recorder = XArm(self.node)
        self.robot_hand_recorder = Tilburg(self.node)

        self.camera_recorder = IntelRealSense(self.node)
        self.digit_360_recorder = Digit360(self.node)
        self.hand_publisher = self.node.create_publisher(Float64MultiArray, "/th_right/joint_targets", 10)
        self.arm_publisher = self.node.create_publisher(XArmCommand, "/xarm/command", 10)
        self.hand_names = ['thumb_ip', 'thumb_mcp', 'thumb_abd', 'thumb_cmc', 'index_dip', 'indexp_ip', 'index_mcp', 'index_abd', 'middle_dip', 'middle_pip', 'middle_mcp', 'middle_abd', 'ring_dip', 'ring_pip', 'ring_mcp', 'ring_abd']
        self.arm_names = [f"joint_{i}" for i in range(1, 8)]

    def get_qpos(self):
        qpos = {}
        hand_pos = self.robot_hand_recorder.get_qpos()
        for key in self.hand_names:
            qpos[key] = 0.0
        if hand_pos is not None:
            for i in range(len(self.hand_names)):
                qpos[self.hand_names[i]] = hand_pos[i]

        arm_pos = self.robot_arm_recorder.get_qpos()
        for key in self.arm_names:
            qpos[key] = 0.0
        if arm_pos is not None:
            for i in range(len(self.arm_names)):
                qpos[self.arm_names[i]] = arm_pos[i]

        return qpos

    def get_qvel(self):
        arm_qvel = self.robot_arm_recorder.get_qvel()
        hand_qvel = self.robot_hand_recorder.get_qvel()
        out = hand_qvel if hand_qvel is not None else [0.0 for i in range(16)]
        if arm_qvel is None:
            for i in range(9):
                out.append(0.0)
        else:
            out += arm_qvel
            out.append(0.0)
            out.append(0.0)

        return np.array(out)

    def get_effort(self):
        arm_effort = self.robot_arm_recorder.get_effort()
        hand_effort = self.robot_hand_recorder.get_effort()
        out = hand_effort if hand_effort is not None else [0.0 for i in range(16)]
        if arm_effort is None:
            for i in range(9):
                out.append(0.0)
        else:
            out += arm_effort
            out.append(0.0)
            out.append(0.0)

        return np.array(out)

    def get_images(self):
        digit_360_dict = self.digit_360_recorder.get_images()
        camera_dict = self.camera_recorder.get_images()
        out = digit_360_dict | camera_dict
        return out

    def _move_hand(self, desired_action):
        out = Float64MultiArray()
        out.data = desired_action
        self.hand_publisher.publish(out)

    def _move_arm(self, desired_action):
        out = XArmCommand()
        out.pos_x = desired_action[0]
        out.pos_y = desired_action[1]
        out.pos_z = desired_action[2] if desired_action[2] >= 160.0 else 160.0
        out.rot_x = desired_action[3]
        out.rot_y = desired_action[4]
        out.rot_z = desired_action[5]
        out.speed = 100.0
        out.acc = 500.0
        out.mvtime = 0.0
        self.arm_publisher.publish(out)

    def is_ready(self):
        return self.camera_recorder.is_ready() and self.digit_360_recorder.is_ready()

    def _reset_arm(self):
        out = XArmCommand()
        out.pos_x = 500.0
        out.pos_y = 0.0
        out.pos_z = 370.0
        out.rot_x = math.pi
        out.rot_y = - math.pi / 2
        out.rot_z = 0.0
        out.speed = 100.0
        out.acc = 500.0
        out.mvtime = 0.0
        self.arm_publisher.publish(out)

    def _reset_hand(self, num):
        if num == 1:
            qpos = self.robot_hand_recorder.get_qpos()
            desired_action = [0.0, 0.0, 60.0, 30.0]
            for i in range(12):
                desired_action.append(0.0)
            for i in range(16):
                desired_action[i] = abs((desired_action[i] - qpos[i]) / 2)
            out = Float64MultiArray()
            out.data = desired_action
            self.hand_publisher.publish(out)
        if num == 2:
            desired_action = [0.0, 0.0, 60.0, 30.0]
            for i in range(12):
                desired_action.append(0.0)
            out = Float64MultiArray()
            out.data = desired_action
            self.hand_publisher.publish(out)

    def get_observation(self):
        obs = collections.OrderedDict()
        qpos = self.get_qpos()
        obs["qpos"] = list(qpos.values())
        for name in self.hand_names:
            obs[name] = qpos[name]
        for name in self.arm_names:
            obs[name] = qpos[name]
        images = self.get_images()
        for key, value in images.items():
            obs[key] = value
        obs["task"] = "Pick up the box in the middle"
        # Add dummy keys to match dataset feature spec (shape 25, but only 23 joint names)
        obs["dummy_0"] = 0.0
        obs["dummy_1"] = 0.0
        return obs

    def get_reward(self):
        return 0

    def reset(self):
        pass
    
    def reset_robot(self, num):
        self._reset_hand(num)
        self._reset_arm()

    def step(self, action):
        new_action = action
        hand_command = []
        for key in self.hand_names:
            hand_command.append(new_action[key])
        self._move_hand(hand_command)

        arm_command = []
        for key in self.arm_names:
            arm_command.append(new_action[key])
        self._move_arm(arm_command)

def get_action():
    return None

def make_real_env(node):
    env = RealEnv(node)
    return env

def test_real_teleop():
    pass

if __name__ == '__main__':
    test_real_teleop()
