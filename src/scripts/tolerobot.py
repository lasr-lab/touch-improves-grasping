import argparse
import numpy as np
import cv2
import zlib
import csv
from pathlib import Path
from tqdm import tqdm
from rosbags.highlevel import AnyReader
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image as PIL_IMAGE
import os


TARGET_FPS = 30  # Target frame rate for the dataset
MAX_DEPTH_MM = 2500.0  # Maximum interaction distance for depth scaling

global_height = 480
global_width = 848
small_size = 224

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


def decode_depth_to_rgb(image, width=global_width, height=global_height):
    """Decompress zlib bytes and scale to a smooth 3-channel 8-bit image."""
    try:
        depth_mm = np.array(image)
        depth_clipped = np.clip(depth_mm, 0, MAX_DEPTH_MM)
        depth_scaled = (depth_clipped / MAX_DEPTH_MM * 255.0).astype(np.uint8)
        rgb = np.stack([depth_scaled, depth_scaled, depth_scaled], axis=-1)
        img = PIL_IMAGE.fromarray(rgb, 'RGB')
        resized_image = img.resize((global_width, global_height))
        cropped_image = resized_image.crop(((global_width - global_height) / 2, 0, global_width - (global_width - global_height) / 2, global_height))
        final_image = cropped_image.resize((small_size, small_size))
        return final_image

    except Exception as e:
        print(f"Depth decompression error: {e}")
        return None

def process_digit_360(image):
    resized_image = image.resize((small_size, small_size))
    return resized_image

def process_single_episode(bag_path, dataset):
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

    rgb_path = f"{bag_path}/{FOLDERS["rgb"]}/"
    depth_path = f"{bag_path}/{FOLDERS["depth"]}/"
    touch_path = f"{bag_path}/{FOLDERS['touch']}/"
    state_path = f"{bag_path}/{FOLDERS['state']}/"
    action_path = f"{bag_path}/{FOLDERS['action']}/"
    qpos_file = open(f"{state_path}qpos.csv", "r")
    qvel_file = open(f"{state_path}qvel.csv", "r")
    action_file = open(f"{action_path}action.csv", "r")
    qpos = list(csv.reader(qpos_file))[1:]
    qvel = list(csv.reader(qvel_file))[1:]
    action = list(csv.reader(action_file))[1:]
    LENGTH = len(os.listdir(rgb_path))
    # print(LENGTH)

    for frame_count in range(LENGTH):

        # --- Update State ---
        img = PIL_IMAGE.open(f"{rgb_path}{frame_count}.jpg")
        resized_image = img.resize((global_width, global_height))
        cropped_image = resized_image.crop(((global_width - global_height) / 2, 0, global_width - (global_width - global_height) / 2, global_height))
        final_image = cropped_image.resize((small_size, small_size))
        last_data["rgb"] = final_image

        image = PIL_IMAGE.open(f"{depth_path}{frame_count}.png")
        depth_rgb = decode_depth_to_rgb(image)
        if depth_rgb is not None:
            last_data["depth"] = depth_rgb
    
        d0 = PIL_IMAGE.open(f"{touch_path}0/frames/{frame_count}.jpg")
        resized_image = d0.resize((small_size, small_size))
        last_data["digit0"] = resized_image

        d1 = PIL_IMAGE.open(f"{touch_path}1/frames/{frame_count}.jpg")
        resized_image = d1.resize((small_size, small_size))
        last_data["digit1"] = resized_image

        d2 = PIL_IMAGE.open(f"{touch_path}2/frames/{frame_count}.jpg")
        resized_image = d2.resize((small_size, small_size))
        last_data["digit2"] = resized_image

        d3 = PIL_IMAGE.open(f"{touch_path}3/frames/{frame_count}.jpg")
        resized_image = d3.resize((small_size, small_size))
        last_data["digit3"] = resized_image

        last_data["qpos"] = np.array(qpos[frame_count], dtype=np.float32)

        last_data["qvel"] = np.array(qvel[frame_count], dtype=np.float32)

        edit_action = np.array(action[frame_count], dtype=np.float32)
        last_data["action"] = edit_action

        dataset.add_frame(
            {
                "observation.images.realsense_image_raw": last_data["rgb"],
                "observation.images.realsense_image_depth": last_data["depth"],
                "observation.images.digit_rgb_image_0": last_data["digit0"],
                "observation.images.digit_rgb_image_1": last_data["digit1"],
                "observation.images.digit_rgb_image_2": last_data["digit2"],
                "observation.images.digit_rgb_image_3": last_data["digit3"],
                "observation.state": last_data["qpos"],
                "observation.qvel": last_data["qvel"],
                "action": last_data["action"],
                "task": "unspecified"
            }
        )

    return LENGTH


def find_all_success_episodes(root_path):
    """Recursively find all 'success' folders and return all episode paths within them."""
    episodes = []
    root = Path(root_path)
    
    # Walk through the directory tree
    for success_dir in root.rglob("success"):
        if success_dir.is_dir():
            # Each subdirectory within 'success' is an episode
            for episode_dir in success_dir.iterdir():
                if episode_dir.is_dir():
                    episodes.append(episode_dir)
    
    # Sort episodes for consistent ordering
    episodes.sort(key=lambda p: (p.parent.parent.name, p.parent.parent.parent.name, int(p.name) if p.name.isdigit() else p.name))
    return episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to folder to recursively search for 'success' episodes",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root folder where dataset will be saved",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="h1_teleop_data",
        help="Name of the dataset folder",
    )
    args = parser.parse_args()

    output_path = Path(args.output_dir) / args.repo_id
    if output_path.exists():
        print(f"Warning: Output folder '{output_path}' already exists.")


    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=args.output_dir,
        fps=TARGET_FPS,
        features={
            "observation.images.realsense_image_raw": {
                "dtype": "video",
                "shape": (small_size, small_size, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.realsense_image_depth": {
                "dtype": "video",
                "shape": (small_size, small_size, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.digit_rgb_image_0": {
                "dtype": "video",
                "shape": (small_size, small_size, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.digit_rgb_image_1": {
                "dtype": "video",
                "shape": (small_size, small_size, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.digit_rgb_image_2": {
                "dtype": "video",
                "shape": (small_size, small_size, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.digit_rgb_image_3": {
                "dtype": "video",
                "shape": (small_size, small_size, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (23,),
                "names": JOINT_ARRAY,
            },
            "observation.qvel": {
                "dtype": "float32",
                "shape": (23,),
                "names": JOINT_ARRAY,
            },
            "action": {
                "dtype": "float32",
                "shape": (23,),
                "names": JOINT_ARRAY
            },
            "episode_index": {
                "dtype": "int64",
                "shape": (1,),
                "names": ["episode_index"],
            },
            "frame_index": {"dtype": "int64", "shape": (1,), "names": ["frame_index"]},
        },
    )

    # Find all success episodes recursively
    input_path = Path(args.input_dir)
    episodes = find_all_success_episodes(input_path)
    print(f"Found {len(episodes)} successful episodes across all subfolders.")

    total_frames = 0
    for episode_path in tqdm(episodes):
        frames = 0
        try:
            frames = process_single_episode(str(episode_path), dataset)
        except Exception as e:
            with open("failures.txt", "a") as f:
                f.write(f"{str(episode_path)}\n")
        if frames > 0:
            print(f"Frames of episode are: {frames}")
            dataset.save_episode()
        total_frames += frames


if __name__ == "__main__":
    main()
