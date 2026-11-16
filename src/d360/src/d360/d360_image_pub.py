# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
cv2.setNumThreads(4)

class CameraPublisher(Node):
    def __init__(self) -> None:
        super().__init__('d360_image_node')
        self.bridge = CvBridge()

        self.declare_parameter('devices', rclpy.Parameter.Type.STRING_ARRAY)
        self.devices = self.get_parameter("devices").value
        self.caps = []
        self.publisher_compresseds = []
        self.publisher_images = []
        self.threads = []
        for i in range(len(self.devices)):
            device = self.devices[i]
            self.publisher_images.append(self.create_publisher(Image, f'/image_raw/index_{i}', 1))
            self.publisher_compresseds.append(self.create_publisher(CompressedImage, f'/image_raw/compressed/index_{i}', 1))
            self.caps.append(cv2.VideoCapture(device))
            ret, frame = self.caps[i].read()
            # assert ret, f"Failed to open DIGIT360 camera image capture system *{device}*"

            thread = threading.Thread(target=self.publish_camera_image, kwargs={"index": i})
            thread.start()
            self.threads.append(thread)

    def publish_camera_image(self, index: int) -> None:
        while rclpy.ok():
            ret, frame = self.caps[index].read()
            if ret:
                if self.publisher_images[index].get_subscription_count():
                    image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="passthrough")
                    image_msg.header.stamp = self.get_clock().now().to_msg()
                    self.publisher_images[index].publish(image_msg)
                if self.publisher_compresseds[index].get_subscription_count():
                    cimg_msg = self.bridge.cv2_to_compressed_imgmsg(frame, dst_format="jpeg")
                    cimg_msg.header.stamp = self.get_clock().now().to_msg()
                    self.publisher_compresseds[index].publish(cimg_msg)
            else:
                self.get_logger().error(f"Failed to read frame from d360.digit360 camera image capture system, index {index}", throttle_duration_sec=10)
                self.caps[index].release()
                self.caps[index] = cv2.VideoCapture(self.devices[index])
        # self.threads[index].join()

def main(args: list=None) -> None:
    rclpy.init(args=args)

    camera_publisher = CameraPublisher()

    rclpy.spin(camera_publisher)

    camera_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
