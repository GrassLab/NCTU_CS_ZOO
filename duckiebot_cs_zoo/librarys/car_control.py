import time
from typing import Tuple

try:
    from utils.kinematics import (
        Kinematics,
        PoseEstimator,
    )
    from drivers.dagu_wheels_driver import DaguWheelsDriver
except ImportError:
    from ..utils.kinematics import (
        Kinematics,
        PoseEstimator,
    )
    from ..drivers.dagu_wheels_driver import DaguWheelsDriver


class CarControl:
    motor_driver: DaguWheelsDriver = None
    kinematics: Kinematics = None
    pose_estimator: PoseEstimator = None

    def __init__(self, trim_val: float = 0.0):
        self.motor_driver = DaguWheelsDriver()
        self.kinematics = Kinematics()
        self.kinematics.trim_val = trim_val
        self.pose_estimator = PoseEstimator()

    def move(self, velocity: float, omega: float):
        """
        TODO: document for student
        """
        left_speed, right_speed = self.kinematics.inverse(
            velocity=velocity,
            omega=omega,
        )
        self.motor_driver.setWheelsSpeed(left=left_speed, right=right_speed)
        self.pose_estimator.estimate(
            time_ns=time.time_ns(),
            velocity=velocity,
            omega=omega,
        )
        return

    def get_pose(self) -> Tuple[int, float, float, float]:
        return self.pose_estimator.get_pose()
