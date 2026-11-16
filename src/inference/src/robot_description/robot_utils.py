import numpy as np
from PIL import Image as PIL_IMAGE

from sensor_msgs.msg import Image, JointState


target_height = 224
target_width = 224
tmp_height = 480
tmp_width = 848


class IntelRealSense():
    rgb_image = np.zeros((target_width, target_height, 3))
    depth_image = np.zeros((target_width, target_height, 3))
    ready = False

    def __init__(self, node):
        self.node = node
        self.subscription_rgb = self.node.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.cb_image_raw,
            3)
        
        self.subscription_depth = self.node.create_subscription(
            Image,
            '/camera/camera/depth/image_rect_raw',
            self.cb_image_depth,
            3)

    def get_images(self):
        image_dict = dict()
        image_dict["realsense_image_raw"] = self.rgb_image
        image_dict["realsense_image_depth"] = self.depth_image
        return image_dict
    
    def cb_image_raw(self, msg):
        self.ready = True
        rgb_array = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        img = PIL_IMAGE.fromarray(rgb_array, 'RGB')
        resized_image = img.resize((tmp_width, tmp_height))

        cropped_image = resized_image.crop(((tmp_width - tmp_height) / 2, 0, tmp_width - (tmp_width - tmp_height) / 2, tmp_height))
        final_image = cropped_image.resize((target_width, target_height))
        self.rgb_image = np.array(final_image)

    def is_ready(self):
        return self.ready

    def cb_image_depth(self, msg):
        MAX_DEPTH_MM = 2500.0  # Maximum interaction distance for depth scaling

        depth_mm = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
        depth_clipped = np.clip(depth_mm, 0, MAX_DEPTH_MM)
        depth_scaled = (depth_clipped / MAX_DEPTH_MM * 255.0).astype(np.uint8)
        rgb = np.stack([depth_scaled, depth_scaled, depth_scaled], axis=-1)
        img = PIL_IMAGE.fromarray(rgb, 'RGB')
        resized_image = img.resize((tmp_width, tmp_height))
        cropped_image = resized_image.crop(((tmp_width - tmp_height) / 2, 0, tmp_width - (tmp_width - tmp_height) / 2, tmp_height))
        final_image = cropped_image.resize((target_height, target_height))
        self.depth_image = np.array(final_image)


class Digit360():
    rgb_image0 = np.zeros((target_width, target_height, 3))
    rgb_image1 = np.zeros((target_width, target_height, 3))
    rgb_image2 = np.zeros((target_width, target_height, 3))
    rgb_image3 = np.zeros((target_width, target_height, 3))

    def __init__(self, node):
        self.node = node
        self._ready = [False, False, False, False]
        self._subs = [
            self.node.create_subscription(Image, '/image_raw/index_0', self.cb_image_raw0, 3),
            self.node.create_subscription(Image, '/image_raw/index_1', self.cb_image_raw1, 3),
            self.node.create_subscription(Image, '/image_raw/index_2', self.cb_image_raw2, 3),
            self.node.create_subscription(Image, '/image_raw/index_3', self.cb_image_raw3, 3),
        ]

    def is_ready(self):
        return all(self._ready)

    def get_images(self):
        image_dict = dict()
        image_dict["digit_rgb_image_0"] = self.rgb_image0
        image_dict["digit_rgb_image_1"] = self.rgb_image1
        image_dict["digit_rgb_image_2"] = self.rgb_image2
        image_dict["digit_rgb_image_3"] = self.rgb_image3
        return image_dict
    
    def process_digit(self, rgb_array):
        rgb_array = rgb_array[100:600, 420:880]
        img = PIL_IMAGE.fromarray(rgb_array, 'RGB')
        resized_image = img.resize((target_width, target_height))
        return np.array(resized_image)
    
    def cb_image_raw0(self, msg):
        rgb_array = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))[:, :, ::-1]
        self.rgb_image0 = self.process_digit(rgb_array)
        self._ready[0] = True
    
    def cb_image_raw1(self, msg):
        rgb_array = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))[:, :, ::-1]
        self.rgb_image1 = self.process_digit(rgb_array)
        self._ready[1] = True


    def cb_image_raw2(self, msg):
        rgb_array = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))[:, :, ::-1]
        self.rgb_image2 = self.process_digit(rgb_array)
        self._ready[2] = True


    def cb_image_raw3(self, msg):
        rgb_array = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))[:, :, ::-1]
        self.rgb_image3 = self.process_digit(rgb_array)
        self._ready[3] = True


class XArm():
    position = None
    velocity = None
    effort = None

    def __init__(self, node):
        self.node = node
        self.subscription_robot_states = self.node.create_subscription(
            JointState,
            '/xarm/joint_states',
            self.cb_robot_states,
            3)

    def get_effort(self):
        return self.effort
    
    def get_qvel(self):
        return self.velocity
    
    def get_qpos(self):
        return self.position
    
    def cb_robot_states(self, msg):
        self.position = msg.position
        self.velocity = msg.velocity
        self.effort = msg.effort


class Tilburg():
    position = None
    velocity = None
    effort = None

    def __init__(self, node):
        self.node = node
        self.subscription_robot_states = self.node.create_subscription(
            JointState,
            '/th_right/joint_states',
            self.cb_hand_states,
            3)

    def get_effort(self):
        return self.effort
    
    def get_qvel(self):
        return self.velocity
    
    def get_qpos(self):
        return self.position
    
    def cb_hand_states(self, msg):
        self.position = msg.position
        self.velocity = msg.velocity
        self.effort = msg.effort