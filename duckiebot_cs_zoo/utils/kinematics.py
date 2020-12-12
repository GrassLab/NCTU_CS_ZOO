from typing import Tuple
import time

import numpy as np


class Kinematics:
    """
    Configuration:
    ~gain (:obj:`float`): scaling factor applied to the desired velocity, default is 1.0
    ~trim (:obj:`float`): trimming factor that is typically used to offset differences in the
        behaviour of the left and right motors, it is recommended to use a value that results in
        the robot moving in a straight line when forward command is given, default is 0.0
    ~baseline (:obj:`float`): the distance between the two wheels of the robot, default is 0.1
    ~radius (:obj:`float`): radius of the wheel, default is 0.0318
    ~k (:obj:`float`): motor constant, assumed equal for both motors, default is 27.0
    ~limit (:obj:`float`): limits the final commands sent to the motors, default is 1.0
    """
    gain_val: float = 1.0  # editable
    trim_val: float = 0.0  # editable
    baseline: float = 0.1
    radius: float = 0.0318
    k: float = 27.0
    limit_val: float = 1.0  # editable

    def get_k_inv(self) -> Tuple[float, float]:
        # assuming same motor constants k for both motors
        k_r = self.k
        k_l = self.k

        # adjusting k by gain and trim
        k_r_inv = (self.gain_val + self.trim_val) / k_r
        k_l_inv = (self.gain_val - self.trim_val) / k_l

        return (k_l_inv, k_r_inv)

    def inverse(self, velocity: float, omega: float) -> Tuple[float, float]:
        """INVERSE KINEMATICS PART
        return left_motor, right_motor
        """
        k_l_inv, k_r_inv = self.get_k_inv()

        omega_r = (velocity + 0.5 * omega * self.baseline) / self.radius
        omega_l = (velocity - 0.5 * omega * self.baseline) / self.radius

        # conversion from motor rotation rate to duty cycle
        u_r = omega_r * k_r_inv
        u_l = omega_l * k_l_inv

        if self.limit_val is not None:
            u_r = self.trim(u_r, -self.limit_val, self.limit_val)
            u_l = self.trim(u_l, -self.limit_val, self.limit_val)

        return (u_l, u_r)

    def forward(
        self,
        left_motor: float, right_motor: float,
    ) -> Tuple[float, float]:
        """FORWARD KINEMATICS PART
        return velocity, omega
        """
        k_l_inv, k_r_inv = self.get_k_inv()

        # Conversion from motor duty to motor rotation rate
        omega_r = right_motor / k_r_inv
        omega_l = left_motor / k_l_inv

        # Compute linear and angular velocity of the platform
        velocity = (self.radius * omega_r + self.radius * omega_l) / 2.0
        omega = (self.radius * omega_r - self.radius * omega_l) / self.baseline

        return (velocity, omega)

    @staticmethod
    def trim(value, low, high):
        """
        Trims a value to be between some bounds.

        Args:
            value: the value to be trimmed
            low: the minimum bound
            high: the maximum bound

        Returns:
            the trimmed value
        """
        return max(min(value, high), low)


def ns2sec(x): return x / 1e9


class PoseEstimator:
    last_omega: float = None
    last_velocity: float = None
    # time with ns, x, y, theta
    last_pose: Tuple[int, float, float, float] = None

    def __init__(self):
        self.last_pose = (time.time_ns(), 0, 0, 0)

    def estimate(
        self,
        time_ns: int, velocity: float, omega: float,
    ) -> Tuple[int, float, float, float]:
        if self.last_velocity is None or self.last_omega is None:
            self.last_velocity = velocity
            self.last_omega = omega
            return self.last_pose

        dt = ns2sec(time_ns - self.last_pose[0])
        # Integrate the relative movement between the last pose and the current
        theta_delta = self.last_omega * dt
        # to ensure no division by zero for radius calculation
        if np.abs(self.last_omega) < 0.000001:
            # straight line
            x_delta = self.last_velocity * dt
            y_delta = 0
        else:
            # arc of circle
            radius = self.last_velocity / self.last_omega
            x_delta = radius * np.sin(theta_delta)
            y_delta = radius * (1.0 - np.cos(theta_delta))

        # Add to the previous to get absolute pose relative to the starting
        # position
        theta_res = self.last_pose[3] + theta_delta
        x_res = (
            self.last_pose[1]
            + x_delta * np.cos(self.last_pose[3])
            - y_delta * np.sin(self.last_pose[3])
        )
        y_res = (
            self.last_pose[2]
            + y_delta * np.cos(self.last_pose[3])
            + x_delta * np.sin(self.last_pose[3])
        )

        # Update the stored last pose
        self.last_pose = (time_ns, x_res, y_res, theta_res)

        self.last_velocity = velocity
        self.last_omega = omega

        return self.last_pose

    def get_pose(self) -> Tuple[int, float, float, float]:
        return self.last_pose
