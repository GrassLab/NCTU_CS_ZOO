from drivers.dagu_wheels_driver import DaguWheelsDriver
from msg_types import MotorCmdStamp, Twist2DStamp
import numpy as np
import time


class KinematicsCaculator(object):
    def __init__(self):
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
        ~v_max (:obj:`float`): limits the input velocity, default is 1.0
        ~omega_max (:obj:`float`): limits the input steering angle, default is 8.0
        """
        self.gain_val = 1.0  # editable
        self.trim_val = 0  # editable
        self.baseline = 0.1
        self.radius = 0.0318
        self.k = 27.0
        self.limit_val = 1.0  # editable
        self.v_max = 1.0
        self.omega_max = 8.0

    def transfer(self, twist_obj):
        # INVERSE KINEMATICS PART

        # trim the desired commands such that they are within the limits:
        twist_obj.v = self.trim(
            twist_obj.v,
            low=-self.v_max,
            high=self.v_max
        )
        twist_obj.omega = self.trim(
            twist_obj.omega,
            low=-self.omega_max,
            high=self.omega_max
        )

        # assuming same motor constants k for both motors
        k_r = self.k
        k_l = self.k

        # adjusting k by gain and trim
        k_r_inv = (self.gain_val + self.trim_val) / k_r
        k_l_inv = (self.gain_val - self.trim_val) / k_l

        omega_r = (twist_obj.v + 0.5 * twist_obj.omega * self.baseline) / self.radius
        omega_l = (twist_obj.v - 0.5 * twist_obj.omega * self.baseline) / self.radius

        # conversion from motor rotation rate to duty cycle
        # u_r = (gain + trim) (v + 0.5 * omega * b) / (r * k_r)
        u_r = omega_r * k_r_inv
        # u_l = (gain - trim) (v - 0.5 * omega * b) / (r * k_l)
        u_l = omega_l * k_l_inv

        # limiting output to limit, which is 1.0 for the duckiebot
        u_r_limited = self.trim(u_r, -self.limit_val, self.limit_val)
        u_l_limited = self.trim(u_l, -self.limit_val, self.limit_val)

        # FORWARD KINEMATICS PART

        # Conversion from motor duty to motor rotation rate
        omega_r = u_r_limited / k_r_inv
        omega_l = u_l_limited / k_l_inv

        # Compute linear and angular velocity of the platform
        v = (self.radius * omega_r + self.radius * omega_l) / 2.0
        omega = (self.radius * omega_r - self.radius * omega_l) / self.baseline

        return Twist2DStamp(None, omega, v), MotorCmdStamp(None, u_l_limited, u_r_limited)

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


class MotorControl(object):

    def __init__(self):
        self.motor_driver = DaguWheelsDriver()

    def move(self, motor_obj):
        self.motor_driver.setWheelsSpeed(motor_obj.left_speed, motor_obj.right_speed)
        print('Set motor speed: left={},right={}'.format(motor_obj.left_speed, motor_obj.right_speed))


motor_control = MotorControl()
kinematics_cal = KinematicsCaculator()


omega = -4
velocity = 0.5
cmd = Twist2DStamp(time.time_ns(), omega, velocity)
now_twist, motor_cmd = kinematics_cal.transfer(cmd)
cur_time = time.time_ns()
now_twist.time_ns = cur_time
motor_cmd.time_ns = cur_time
motor_control.move(motor_cmd)

