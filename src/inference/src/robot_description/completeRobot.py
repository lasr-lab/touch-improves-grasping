from lerobot.robots.robot import Robot
from .environment import make_real_env
from .config_robot import CompleteRobotConfig
import rclpy
from typing import Any


class CompleteRobot(Robot):
    name = "complete_robot"
    config_class = CompleteRobotConfig
    def __init__(self, config: CompleteRobotConfig):
        super().__init__(config)
        rclpy.init()
        self.node = rclpy.create_node("beso_eval_node")
        self.env = make_real_env(self.node)

    def connect(self, calibrate: bool = True) -> None:
        pass

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        self.env.step(action)

    def spin(self):
        rclpy.spin_once(self.node)
    
    def get_observation(self) -> dict[str, Any]:
        obs = self.env.get_observation()
        return obs
    
    def reset(self):
        self.env.reset_robot(1)

    def reset_2(self):
        self.env.reset_robot(2)

    def disconnect(self) -> None:
        pass

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def is_calibrated(self) -> bool:
        return True
    
    def is_connected(self) -> bool:
        return self.env.is_ready()
    
    def action_features(self) -> dict:
        return {}
    
    @property
    def observation_features(self):
        return {}



