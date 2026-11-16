import os
from functools import partial

import rclpy
from lasr_msgs.srv import ClassifyD360Image
from lasr_msgs.msg import ContactClassification
from sensor_msgs.msg import Image
import numpy as np
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from PIL import Image as PILImage
import torch
import torch.nn as nn
from torchvision import models, transforms


NUM_SENSORS = 4


class D360Classifier(Node):
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]
    INPUT_SIZE = (224, 224)

    eval_transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    def __init__(self):
        super().__init__('d360_classifier')

        default_ckpt = os.path.join(
            get_package_share_directory('d360'), 'checkpoints', 'contact_classifier.pth'
        )
        self.declare_parameter('checkpoint', default_ckpt)
        self.declare_parameter('num_sensors', NUM_SENSORS)

        checkpoint = self.get_parameter('checkpoint').value
        num_sensors = self.get_parameter('num_sensors').value
        print(f"number of sensors: {num_sensors}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # if self.device.type == "cpu":
        #     torch.set_num_threads(2)
        self.ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.model = self.build_model(num_classes=len(self.ckpt["class_names"]), pretrained=False).to(self.device)
        self.model.load_state_dict(self.ckpt["model_state_dict"])
        self.model.eval()

        self._subscribers = []
        self._publishers = []
        for i in range(num_sensors):
            print(f"Current length: {len(self._publishers)}")
            self._subscribers.append(self.create_subscription(
                Image, f"/image_raw/index_{i}", partial(self._image_cb, index=i), 1
            ))
            self.create_publisher(
                ContactClassification, f"/d360/classified/index_{i}", 1
            )
            print(f"length directoy before: {len(self._publishers)}")
            # self._publishers.append(tmp)
        print(f"num_sensors: {num_sensors}, length: {len(self._publishers)}")
        self.create_service(ClassifyD360Image, 'classify_d360_image', self._classify_srv)

    def _image_cb(self, msg: Image, index: int):
        rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        rgb = rgb[100:600, 420:880]
        img = PILImage.fromarray(rgb)
        in_contact, confidence = self.infer(img)
        result = ContactClassification()
        result.in_contact = in_contact
        result.confidence = confidence
        self._publishers[index].publish(result)

    def _classify_srv(self, request, response):
        rgb = np.frombuffer(request.image.data, dtype=np.uint8).reshape(
            (request.image.height, request.image.width, 3)
        )
        rgb = rgb[100:600, 420:880]
        img = PILImage.fromarray(rgb)
        in_contact, confidence = self.infer(img)
        response.in_contact = in_contact
        response.confidence = confidence
        return response

    @torch.no_grad()
    def infer(self, img: PILImage):
        class_names = self.ckpt["class_names"]
        tensor = self.eval_transform(img).unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()

        in_contact = class_names[pred_idx] == "down"
        confidence = probs[1] if in_contact else probs[0]
        return in_contact, float(confidence)

    def build_model(self, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)

        for param in model.features.parameters():
            param.requires_grad = False

        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )
        return model


def main(args=None):
    rclpy.init(args=args)
    node = D360Classifier()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
