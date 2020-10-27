from motor_control import *
from kinematics_calculator import *
from msg_types import Twist2DStamp
import time


motor_control = MotorControl()
kinematics_cal = KinematicsCaculator()


while True:
    key = input("Cmd: ")
    omega = 0
    velocity = 0
    if key == "w":
        omega = 0
        velocity = 0.2
    elif key == "s":
        omega = 0
        velocity = -0.2
    elif key == "a":
        omega = 2
        velocity = 0
    elif key == "d":
        omega = -2
        velocity = 0
    elif key == " ":
        omega = 0
        velocity = 0
    else:
        continue

    cmd = Twist2DStamp(time.time_ns(), omega, velocity)
    now_twist, motor_cmd = kinematics_cal.transfer(cmd)
    cur_time = time.time_ns()
    now_twist.time_ns = cur_time
    motor_cmd.time_ns = cur_time
    motor_control.move(motor_cmd)

