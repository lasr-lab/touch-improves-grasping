#!/usr/bin/env python3
"""
Color Image Compression Node
Compresses color images to JPEG for network transfer.
"""

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image


class ImageThrottleCompressor(Node):
    def __init__(self):
        super().__init__("image_compressor")

        # Parameters
        self.declare_parameter("quality", 50)  # JPEG quality
        self.declare_parameter("input_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("output_topic", "/camera/image_compressed")

        self.quality = int(self.get_parameter("quality").value)
        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value

        self.bridge = CvBridge()

        # QOS: Reliable is required to visualize in Unity/RViz without warnings
        pub_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        self.publisher = self.create_publisher(CompressedImage, self.output_topic, qos_profile=pub_qos)

        # Subscriber QoS
        sub_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        self.subscriber = self.create_subscription(
            Image,
            self.input_topic,
            self._image_callback,
            qos_profile=sub_qos,
        )

        self.get_logger().info(f"Image compressor: {self.input_topic} -> {self.output_topic} quality={self.quality}")

    def _image_callback(self, msg):
        """Process and publish if enough time has passed."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Could not convert image: {e}")
            return

        # Encode as JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        result, encimg = cv2.imencode(".jpg", cv_image, encode_param)

        if not result:
            self.get_logger().error("Could not encode image")
            return

        compressed_msg = CompressedImage()
        compressed_msg.header = msg.header
        compressed_msg.format = "jpeg"
        compressed_msg.data = encimg.tobytes()

        self.publisher.publish(compressed_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ImageThrottleCompressor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
