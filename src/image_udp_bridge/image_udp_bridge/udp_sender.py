#!/usr/bin/env python3
"""
Original version created by Max Ehrlich. This version was adapted and is maintained by Niklas Leukroth.
"""

"""
UDP Sender: forwards ROS topics to Unity over UDP.
Rate limiting and compression are handled by h1_bandwidth_manager.
This node only handles: UDP transport, chunking, and chunk pacing.
"""

import base64
import json
import socket
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, JointState


class UDPSender(Node):
    def __init__(self):
        super().__init__("ros_to_udp_sender")

        # --- Network parameters ---
        self.declare_parameter("target_ip", rclpy.Parameter.Type.STRING)
        self.declare_parameter("target_port", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("chunk_delay_ms", 2.0)

        self.target_ip = self.get_parameter("target_ip").value
        self.target_port = self.get_parameter("target_port").value
        self.chunk_delay_ms = self.get_parameter("chunk_delay_ms").get_parameter_value().double_value

        # Small send buffer to avoid kernel-level bursting over WiFi
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 128 * 1024)

        self.max_udp_size = 40000

        low_latency_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self.color_image_sub = self.create_subscription(
            CompressedImage,
            "/camera/image_compressed",
            self._color_image_callback,
            low_latency_qos,
        )

        self._last_camera_info_send_time = 0.0
        self._camera_info_interval = 5.0
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            "/d435/color/camera_info",
            self._camera_info_callback,
            low_latency_qos,
        )

        self._stats_timer = self.create_timer(10.0, self._log_stats)
        self._msgs_sent = 0
        self._bytes_sent = 0
        self._color_bytes = 0

        self.get_logger().info(
            f"UDP Sender (transport only) -> {self.target_ip}:{self.target_port}\n"
            f"  chunk delay: {self.chunk_delay_ms} ms\n"
            f"  Rate & compression handled by h1_bandwidth_manager"
        )

    def _log_stats(self):
        self.get_logger().info(
            f"[stats] {self._msgs_sent} msgs | {self._bytes_sent / 1024:.0f} KB total | "
            f"color={self._color_bytes / 1024:.0f} KB "
        )
        self._msgs_sent = 0
        self._bytes_sent = 0
        self._color_bytes = 0

    def _send_udp(self, data: dict, msg_type: str):
        try:
            message = {"type": msg_type, "data": data}
            json_data = json.dumps(message).encode("utf-8")

            if len(json_data) > self.max_udp_size:
                self._send_chunked(json_data, msg_type)
            else:
                self.sock.sendto(json_data, (self.target_ip, self.target_port))
                self._bytes_sent += len(json_data)
            self._msgs_sent += 1
        except Exception as e:
            self.get_logger().error(f"UDP send error: {e}")

    def _send_chunked(self, data: bytes, msg_type: str):
        chunk_size = self.max_udp_size - 200
        total_chunks = (len(data) + chunk_size - 1) // chunk_size
        message_id = id(data) & 0xFFFFFFFF

        for i in range(total_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, len(data))

            header = {
                "type": "chunk",
                "msg_type": msg_type,
                "message_id": message_id,
                "chunk_index": i,
                "total_chunks": total_chunks,
                "data": base64.b64encode(data[start:end]).decode("ascii"),
            }
            chunk_json = json.dumps(header).encode("utf-8")
            self.sock.sendto(chunk_json, (self.target_ip, self.target_port))
            self._bytes_sent += len(chunk_json)

            if i < total_chunks - 1:
                time.sleep(self.chunk_delay_ms / 1000.0)

    def _color_image_callback(self, msg: CompressedImage):
        """Forward compressed color image as-is."""
        before = self._bytes_sent
        self._send_udp(
            {
                "format": "jpeg",
                "data": base64.b64encode(bytes(msg.data)).decode("ascii"),
                "timestamp_sec": msg.header.stamp.sec,
                "timestamp_nanosec": msg.header.stamp.nanosec,
                "frame_id": msg.header.frame_id,
            },
            "color_image",
        )
        self._color_bytes += self._bytes_sent - before

    def _camera_info_callback(self, msg: CameraInfo):
        now = time.time()
        if now - self._last_camera_info_send_time < self._camera_info_interval:
            return
        self._last_camera_info_send_time = now

        k = list(msg.k)
        self._send_udp(
            {
                "fx": k[0],
                "fy": k[4],
                "cx": k[2],
                "cy": k[5],
                "width": msg.width,
                "height": msg.height,
            },
            "camera_info",
        )

    def shutdown(self):
        self.sock.close()


def main(args=None):
    rclpy.init(args=args)
    sender = UDPSender()

    sender.get_logger().info("Subscribing to ROS topics:")
    sender.get_logger().info("  - /camera/color/image_compressed")
    sender.get_logger().info("  - /camera_info")

    try:
        rclpy.spin(sender)
    except KeyboardInterrupt:
        sender.get_logger().info("Shutting down...")
    finally:
        sender.shutdown()
        sender.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
