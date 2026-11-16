"""
Synchronized bag recorder — starts ros2 bag record paused, waits for topic
discovery to complete, then resumes so no data is missed.

Usage in deploy.py:
    from bag_recorder import SyncBagRecorder

    recorder = SyncBagRecorder(
        output_dir="./bags/episode_0",
        topic_regex="/act/.*",
        expected_topics=[           # optional — omit to skip the check
            "/act/qpos", "/act/qvel", "/act/action",
            "/act/realsense/image", "/act/realsense/image_depth",
            "/act/digit360/image_0", "/act/digit360/image_1",
            "/act/digit360/image_2", "/act/digit360/image_3",
        ],
        discovery_timeout=10.0,
    )
    recorder.start()      # blocks until all expected topics are discovered
    # ... run inference ...
    recorder.stop()
"""

import os
import re
import subprocess
import time
import signal
import shutil

import rclpy
from rclpy.node import Node
from rosbag2_interfaces.srv import Resume


RECORDER_NODE_NAME = "rosbag2_recorder"


class SyncBagRecorder:
    """Start ros2 bag record paused, wait for discovery, then resume."""

    def __init__(
        self,
        output_dir: str,
        topic_regex: str = "/act/.*",
        expected_topics: list[str] | None = None,
        discovery_timeout: float = 15.0,
        poll_interval: float = 0.2,
        recorder_node_name: str = RECORDER_NODE_NAME,
    ):
        self.output_dir = output_dir
        self.topic_regex = topic_regex
        self.expected_topics = set(expected_topics) if expected_topics else None
        self.discovery_timeout = discovery_timeout
        self.poll_interval = poll_interval
        self.recorder_node_name = recorder_node_name
        self._proc: subprocess.Popen | None = None
        self._node: Node | None = None
        self._rclpy_was_initialized = False
        self._did_init_rclpy = False

        self.dead = False

    @staticmethod
    def _resolve_output_dir(output_dir: str) -> str:
        """If *output_dir* already exists, increment a trailing counter until
        a free path is found (e.g. episode_0 → episode_1 → episode_2)."""
        if not os.path.exists(output_dir):
            return output_dir
        base, num = re.match(r"^(.*?)(\d+)$", output_dir).groups()
        idx = int(num)
        while os.path.exists(f"{base}{idx}"):
            idx += 1
        resolved = f"{base}{idx}"
        print(f"Output dir '{output_dir}' exists — using '{resolved}' instead")
        return resolved

    def start(self) -> bool:
        """Launch the recorder paused, wait for discovery, then resume.

        Blocks until all expected topics are subscribed or timeout is reached.
        Raises RuntimeError on timeout.
        """
        self.output_dir = self._resolve_output_dir(self.output_dir)
        self._proc = subprocess.Popen(
            [
                "ros2", "bag", "record",
                "--start-paused",
                "-o", self.output_dir,
                "-e", self.topic_regex,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        self._rclpy_was_initialized = rclpy.ok()
        if not self._rclpy_was_initialized:
            rclpy.init()
            self._did_init_rclpy = True
        self._node = rclpy.create_node("_bag_recorder_helper")
        try:
            self._wait_for_recorder_node()
            self._wait_for_discovery()
            self._resume()
        except Exception:
            self._dump_proc_output()
            self.stop()
            raise
        if self.dead:
            print("dead")
            os.kill(self._proc.pid, signal.SIGTERM)
            time.sleep(1.5)
            shutil.rmtree(self.output_dir, ignore_errors=True)
            return False
        return True

        
        

    def stop(self):
        """Terminate the recorder process and shut down the helper node."""
        if self._proc and self._proc.poll() is None:
            self._proc.send_signal(signal.SIGINT)
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=2)
        if self._proc and self._proc.stdout:
            try:
                self._proc.stdout.close()
            except Exception:
                pass
        self._proc = None

        if self._node:
            self._node.destroy_node()
            self._node = None
        if not self._rclpy_was_initialized and self._did_init_rclpy:
            try:
                rclpy.shutdown()
            except RuntimeError:
                pass

    # ---- internal ----

    def _dump_proc_output(self):
        """Print captured subprocess output for debugging."""
        if self._proc and self._proc.stdout:
            try:
                output = self._proc.stdout.read()
                if output:
                    print(f"--- ros2 bag record output ---\n{output}--- end ---")
            except Exception:
                pass

    def _wait_for_recorder_node(self, timeout: float = 10.0):
        """Block until the recorder node is visible on the ROS graph."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._proc.poll() is not None:
                self._dump_proc_output()
                raise RuntimeError(
                    f"ros2 bag record exited prematurely with code {self._proc.returncode}. "
                    "Check the output above for the cause."
                )
            names = self._node.get_node_names()
            if self.recorder_node_name in names:
                return
            rclpy.spin_once(self._node, timeout_sec=0.1)
        self._dump_proc_output()
        raise RuntimeError(
            f"Recorder node '{self.recorder_node_name}' not found after {timeout}s. "
            "Is ros2 bag record running?"
        )

    def _get_recorder_subscriptions(self) -> set[str]:
        """Return the set of topics the recorder node is subscribed to."""
        try:
            subs = self._node.get_subscriber_names_and_types_by_node(
                self.recorder_node_name, ""
            )
            return {name for name, _ in subs}
        except Exception:
            return set()

    def _wait_for_discovery(self):
        """Block until all expected topics appear in the recorder's subscriptions."""
        if not self.expected_topics:
            # No expected list — just wait a short grace period for discovery
            time.sleep(1.0)
            return

        deadline = time.time() + self.discovery_timeout
        while time.time() < deadline:
            subscribed = self._get_recorder_subscriptions()
            missing = self.expected_topics - subscribed
            if not missing:
                return
            rclpy.spin_once(self._node, timeout_sec=self.poll_interval)

        subscribed = self._get_recorder_subscriptions()
        missing = self.expected_topics - subscribed
        raise RuntimeError(
            f"Discovery timed out after {self.discovery_timeout}s. "
            f"Missing topics: {sorted(missing)}"
        )

    def _resume(self):
        """Call the Resume service on the recorder node."""
        client = self._node.create_client(Resume, f"/{self.recorder_node_name}/resume")
        if not client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("Resume service not available on recorder node")
        future = client.call_async(Resume.Request())
        deadline = time.time() + 5.0
        while not future.done() and time.time() < deadline:
            rclpy.spin_once(self._node, timeout_sec=0.5)
        if time.time() > deadline:
            self.dead = True
        else:
            self._node.get_logger().info("Bag recorder resumed — all topics discovered")
