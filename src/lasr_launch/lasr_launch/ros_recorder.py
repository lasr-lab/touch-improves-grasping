# #!/usr/bin/env python3

import subprocess
import signal
import os
import time
import shutil
from std_srvs.srv import SetBool

import numpy as np
import cv2
import csv
from pathlib import Path
from tqdm import tqdm
from rosbags.highlevel import AnyReader



TARGET_FPS = 30  # Target frame rate for the dataset
MAX_DEPTH_MM = 2500.0  # Maximum interaction distance for depth scaling

global_height = 480
global_width = 848
small_size = 224
# global_width = 480

FOLDERS = {
    "rgb": "camera/rgb/frames/",
    "depth": "camera/depth/frames/",
    "touch": "digit360/",
    "action": "action/",
    "state": "state/"
}

TOPICS = {
    "rgb": "/act/realsense/image",
    "depth": "/act/realsense/image_depth",
    # "digit0": "/act/digit/rgb_image_0",
    # "digit1": "/act/digit/rgb_image_1",
    # "digit2": "/act/digit/rgb_image_2",
    # "digit3": "/act/digit/rgb_image_3",
    "digit0": "/act/digit360/image_0",
    "digit1": "/act/digit360/image_1",
    "digit2": "/act/digit360/image_2",
    "digit3": "/act/digit360/image_3",
    "qpos": "/act/qpos",
    "qvel": "/act/qvel",
    "action": "/act/action"
}

ACTION_ARRAY = [
    "x_pos_arm",
    "y_pos_arm",
    "z_pos_arm",
    "x_rot_arm",
    "y_rot_arm",
    "z_rot_arm",
    "speed"
]

JOINT_ARRAY = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "joint_7",
    "thumb_ip",
    "thumb_mcp",
    "thumb_abd",
    "thumb_cmc",
    "index_dip",
    "indexp_ip",
    "index_mcp",
    "index_abd",
    "middle_dip",
    "middle_pip",
    "middle_mcp",
    "middle_abd",
    "ring_dip",
    "ring_pip",
    "ring_mcp",
    "ring_abd",

]



def create_dir_if_not(folder_dir, dir_name):
    dir_list = os.listdir(folder_dir)
    if not dir_name in dir_list:
        os.makedirs(folder_dir + "/" + dir_name)

def process_digit_360(msg):
    rgb_array = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
    rgb_array = rgb_array[100:600, 420:880]
    return rgb_array

def process_single_bag(bag_path, target_fps=TARGET_FPS):
    parent_dir = str(bag_path.parent)
    create_dir_if_not("./", parent_dir)
    end_dir = "_".join(str(bag_path.stem).rsplit("_")[1:])
    out_dir = parent_dir + "/" + end_dir + "/"

    create_dir_if_not(parent_dir, end_dir)
    create_dir_if_not(out_dir, f"{FOLDERS["rgb"]}")
    create_dir_if_not(out_dir, f"{FOLDERS["depth"]}")
    create_dir_if_not(out_dir, f"{FOLDERS["action"]}")
    create_dir_if_not(out_dir, f"{FOLDERS["state"]}")

    action_csv = open(f"{out_dir}{FOLDERS['action']}action.csv", "w+")
    action_writer = csv.writer(action_csv)
    action_writer.writerow(ACTION_ARRAY + [JOINT_ARRAY[i] for i in range(7, len(JOINT_ARRAY))])
    qpos_csv = open(f"{out_dir}{FOLDERS['state']}qpos.csv", "w+")
    qpos_writer = csv.writer(qpos_csv)
    qpos_writer.writerow(JOINT_ARRAY)
    qvel_csv = open(f"{out_dir}{FOLDERS['state']}qvel.csv", "w+")
    qvel_writer = csv.writer(qvel_csv)
    qvel_writer.writerow(JOINT_ARRAY)

    for i in range(4):
        create_dir_if_not(out_dir, f"{FOLDERS["touch"]}{i}/frames/")
    
    min_interval_ns = int(1_000_000_000 / target_fps)

    with AnyReader([Path(bag_path)]) as reader:
        connections = [x for x in reader.connections if x.topic in TOPICS.values()]

        last_data = {
            "rgb": None,
            "depth": None,
            "digit0": None,
            "digit1": None,
            "digit2": None,
            "digit3": None,
            "qpos": np.zeros(25, dtype=np.float32),
            "qvel": np.zeros(25, dtype=np.float32),
            "action": np.zeros(25, dtype=np.float32),
        }

        frame_count = 0
        last_save_time = 0
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            topic = connection.topic

            # --- Update State ---
            if topic == TOPICS["rgb"]:
                rgb_array = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))[:, :, ::-1]
                last_data["rgb"] = rgb_array

            elif topic == TOPICS["depth"]:
                depth_mm = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
                last_data["depth"] = depth_mm

            elif topic == TOPICS["digit0"]:
                rgb_array = process_digit_360(msg)
                last_data["digit0"] = rgb_array


            elif topic == TOPICS["digit1"]:
                rgb_array = process_digit_360(msg)
                last_data["digit1"] = rgb_array

            elif topic == TOPICS["digit2"]:
                rgb_array = process_digit_360(msg)
                last_data["digit2"] = rgb_array

            elif topic == TOPICS["digit3"]:
                rgb_array = process_digit_360(msg)
                last_data["digit3"] = rgb_array


            elif topic == TOPICS["qpos"]:
                last_data["qpos"] = np.array(msg.arm_position.tolist() + msg.hand_position.tolist(),
                    dtype=np.float32,
                )

            elif topic == TOPICS["qvel"]:
                last_data["qvel"] = np.array(msg.arm_velocity.tolist() + msg.hand_velocity.tolist(), dtype=np.float32
                )

            elif topic == TOPICS["action"]:
                if (len(msg.hand_command.tolist())) == 16:
                    xarm = [msg.xarm_command.pos_x,msg.xarm_command.pos_y, msg.xarm_command.pos_z, msg.xarm_command.rot_x, msg.xarm_command.rot_y, msg.xarm_command.rot_z, msg.xarm_command.speed]
                    last_data["action"] = np.array(xarm + msg.hand_command.tolist(), dtype=np.float32)

                elif (len(msg.hand_command.tolist())) == 12:
                    xarm = [msg.xarm_command.pos_x,msg.xarm_command.pos_y, msg.xarm_command.pos_z, msg.xarm_command.rot_x, msg.xarm_command.rot_y, msg.xarm_command.rot_z, msg.xarm_command.speed]
                    last_data["action"] = np.array(xarm + msg.hand_command.tolist() + [0.0, 0.0, 0.0, 0.0], dtype=np.float32)


            if topic == TOPICS["rgb"]:
                valid = True
                for val in last_data.values():
                    if val is None:
                        valid = False

                if not valid:
                    continue

                if (timestamp - last_save_time) >= min_interval_ns:

                    cv2.imwrite(f"{out_dir}{FOLDERS["rgb"]}{frame_count}.jpg", last_data["rgb"])
                    cv2.imwrite(f"{out_dir}{FOLDERS["depth"]}{frame_count}.png", last_data["depth"])
                    
                    for i in range(4):
                        cv2.imwrite(f"{out_dir}{FOLDERS["touch"]}{i}/frames/{frame_count}.jpg", last_data[f"digit{i}"])

                    action_writer.writerow(last_data["action"])
                    qpos_writer.writerow(last_data["qpos"])
                    qvel_writer.writerow(last_data["qvel"])

                    frame_count += 1
                    last_save_time = timestamp

        return frame_count


def convertBag(input_dir):
    try:
        input_path = Path(input_dir)
        ros_files = list(input_path.rglob("*.mcap"))
        print(f"Found {len(ros_files)} bags.")

        total_frames = 0
        for bag_path in tqdm(ros_files):
            in_path = Path("".join([i + "/" for i in str(bag_path).rsplit("/")[:-1]]))
            frames = process_single_bag(in_path, target_fps=TARGET_FPS)
            if frames > 0:
                print(f"Frames of episode are: {frames}")
            total_frames += frames
    finally:
        print("Episode converted")



import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool as BoolMsg

last_state = False
p = None

success_name = "success"
failure_name = "failure"
folder_dir = "./"

def create_dir_if_not(folder_dir, dir_name):
    dir_list = os.listdir(folder_dir)
    if not dir_name in dir_list:
        os.makedirs(folder_dir + "/" + dir_name)

def look_through_dir(folder_dir, curr_string, is_success):
    global x
    dir_list = os.listdir(folder_dir + "/" + curr_string)
    for string in dir_list:
        if string in [success_name, failure_name]:
            continue
        if os.path.isdir(string):
            current_folder_dir, is_mcap = look_through_dir(folder_dir, string, is_success)
            if is_mcap:
                move(folder_dir, current_folder_dir, is_success)
        elif ".mcap" in string:
            return folder_dir + curr_string, True

    return None, False  

def move(base_folder, folder_dir, is_success):
    if is_success:
        out_path = base_folder + success_name + "/"
        shutil.move(folder_dir, out_path)
        print(f"Moved {folder_dir} to {success_name}")
        convertBag(out_path + folder_dir)
    else:
        out_path = base_folder + failure_name + "/"
        shutil.move(folder_dir, out_path) 
        print(f"Moved {folder_dir} to {failure_name}")
        convertBag(out_path + folder_dir)


class RoslogHandlerNode(Node):

    def __init__(self):
        super().__init__('ros_recorder')
        self.subscription = self.create_subscription(
            BoolMsg,
            '/capture',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.srv = self.create_service(SetBool, 'process_bag', self.process_bag)

    def process_bag(self, request, response):
        look_through_dir(folder_dir, "", request.data)
        response.success = True
        return response



    def listener_callback(self, msg):
        global last_state, p
        state = msg.data
        if ((last_state == False) and (state == True)):
            p = subprocess.Popen(
                ["ros2", "bag", "record", "-e", "/lasr/*"])
            self.get_logger().info("Started recording bagfile")
        if ((last_state == True) and (state == False)):
            os.kill(p.pid, signal.SIGTERM)
            time.sleep(1.5)
            if not p.poll():
                self.get_logger().info("Finished recording bagfile")
            else:
                self.get_logger().info("Ending recording of log bagfile has failed")

        last_state = state
        return


def main(args=None):
    rclpy.init(args=args)

    # check if subfolder has success and failure structure, else create it
    create_dir_if_not(folder_dir, success_name)
    create_dir_if_not(folder_dir, failure_name)

    minimal_subscriber = RoslogHandlerNode()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
