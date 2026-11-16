import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm


TARGET_FPS = 30  # Must match the frame rate processMCAP.py saved the frames at
CRF = 23  # libx264 quality, lower is better
PRESET = "veryfast"

RGB_PARENT = ("camera", "rgb")
FRAMES_DIR_NAME = "frames"
VIDEO_NAME = "rgb.mp4"

# Directories that can never contain camera/rgb/frames, skipped to keep the walk fast.
SKIP_DIR_NAMES = {"action", "state", "digit360", FRAMES_DIR_NAME}


def find_frames_dirs(root):
    """Return every <episode>/camera/rgb/frames directory under root, naturally sorted."""
    found = []
    for dirpath, dirnames, _ in os.walk(root):
        if Path(dirpath).parts[-2:] == RGB_PARENT and FRAMES_DIR_NAME in dirnames:
            found.append(Path(dirpath) / FRAMES_DIR_NAME)
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]

    def sort_key(frames_dir):
        episode_dir = frames_dir.parent.parent.parent
        return tuple(
            (0, int(part)) if part.isdigit() else (1, part)
            for part in episode_dir.relative_to(root).parts
        )

    return sorted(found, key=sort_key)


def list_frame_numbers(frames_dir):
    """Return the sorted integer stems of every <n>.jpg frame in frames_dir."""
    numbers = []
    for name in os.listdir(frames_dir):
        stem, _, ext = name.rpartition(".")
        if ext.lower() == "jpg" and stem.isdigit():
            numbers.append(int(stem))
    return sorted(numbers)


def encode_episode(frames_dir, video_path, fps, crf, overwrite):
    """Encode one episode's frames into video_path. Returns (status, detail)."""
    if video_path.exists() and not overwrite:
        return "skipped", "video already exists"

    numbers = list_frame_numbers(frames_dir)
    if not numbers:
        return "failed", "no .jpg frames found"

    # The %d.jpg input pattern stops at the first missing index, which would silently
    # truncate the video, so require a gapless run starting at 0.
    if numbers != list(range(len(numbers))):
        present = set(numbers)
        missing = next(i for i in range(len(numbers)) if i not in present)
        return "failed", f"frame numbering is not gapless from 0 (first gap at {missing})"

    # Keep the .mp4 extension on the temporary file so ffmpeg still infers the muxer.
    tmp_path = video_path.with_name(f"{video_path.stem}.part{video_path.suffix}")
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-framerate", str(fps),
        "-pattern_type", "sequence",
        "-start_number", str(numbers[0]),
        "-i", str(frames_dir / "%d.jpg"),
        "-c:v", "libx264",
        "-preset", PRESET,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        str(tmp_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tmp_path.unlink(missing_ok=True)
        return "failed", result.stderr.strip().splitlines()[-1] if result.stderr.strip() else f"ffmpeg exited {result.returncode}"

    os.replace(tmp_path, video_path)
    return "encoded", f"{len(numbers)} frames"


def main():
    parser = argparse.ArgumentParser(
        description="Create an rgb.mp4 video from the saved RGB frames of every episode."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=".",
        help="Root folder to search recursively for camera/rgb/frames directories",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=TARGET_FPS,
        help="Frame rate of the source frames and the resulting video",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=CRF,
        help="libx264 constant rate factor, lower means higher quality",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-encode episodes that already contain a video",
    )
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        print("Error: ffmpeg was not found on PATH.")
        sys.exit(1)

    root = Path(args.input_dir).resolve()
    if not root.is_dir():
        print(f"Error: Input folder '{root}' does not exist.")
        sys.exit(1)

    print(f"Scanning '{root}' for RGB frame folders...")
    frames_dirs = find_frames_dirs(root)
    print(f"Found {len(frames_dirs)} episodes.")
    if not frames_dirs:
        return

    counts = {"encoded": 0, "skipped": 0, "failed": 0}
    failures = []

    with tqdm(frames_dirs, unit="episode") as pbar:
        for frames_dir in pbar:
            episode_dir = frames_dir.parent.parent.parent
            video_path = frames_dir.parent / VIDEO_NAME
            status, detail = encode_episode(frames_dir, video_path, args.fps, args.crf, args.overwrite)
            counts[status] += 1
            if status == "failed":
                failures.append((episode_dir, detail))
                # tqdm.write keeps failures readable instead of shredding the bar.
                tqdm.write(f"[failed] {episode_dir.relative_to(root)}: {detail}")
            pbar.set_postfix_str(
                f"encoded={counts['encoded']} skipped={counts['skipped']} failed={counts['failed']}"
            )

    print(
        f"\nDone. Encoded: {counts['encoded']}, "
        f"Skipped: {counts['skipped']}, Failed: {counts['failed']}"
    )
    if failures:
        print("\nFailed episodes:")
        for episode_dir, detail in failures:
            print(f"  {episode_dir.relative_to(root)}: {detail}")


if __name__ == "__main__":
    main()
