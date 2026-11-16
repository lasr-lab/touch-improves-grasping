# scripts

Utility scripts for dataset processing, annotation, and visualization.

## Scripts

| Script | Description |
|--------|-------------|
| `tolerobot.py` | Converts ROS 2 bag data into LeRobot dataset format — extracts RGB, depth, Digit 360 tactile images, actions, and joint states at 30 FPS |
| `annotate.py` | Interactive GUI annotator for episode videos — plays RGB videos and sorts episodes into success/failure/delete categories with undo support and a CSV audit log |
| `combineDatasets.py` | Merges two LeRobot datasets (e.g., side-grasp + top-grasp) into a single combined dataset |
| `combine.bash` | Shell script wrapper for dataset combination |
| `createVideos.py` | Batch-encodes RGB frame sequences into H.264 MP4 videos using ffmpeg |

## Usage

### Convert bags to LeRobot format

```bash
python3 tolerobot.py
```

Reads ROS 2 bags, extracts synchronized topics (`/act/realsense/image`, `/act/digit360/image_*`, `/act/qpos`, etc.), and writes a LeRobot-compatible dataset with RGB, depth, tactile, action, and state folders.

### Annotate episodes

```bash
python3 annotate.py                       # annotate everything under cwd
python3 annotate.py --root act/nontac     # one group only
python3 annotate.py --scan-only           # print what was found and exit
python3 annotate.py --dry-run             # play + decide, move nothing
```

### Combine datasets

```bash
python3 combineDatasets.py
```

### Create episode videos

```bash
python3 createVideos.py
```

Walks the dataset tree, finds `camera/rgb/frames/` directories, and encodes each into `rgb.mp4` at 30 FPS.

## Dependencies

- `numpy`, `opencv-python`
- `rosbags` (for reading ROS 2 bags)
- `lerobot` (for LeRobot dataset format)
- `Pillow`, `tqdm`
- `ffmpeg` (for video creation)
