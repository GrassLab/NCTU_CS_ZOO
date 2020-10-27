class MotorCmdStamp:
    def __init__(self, time_ns, left_speed, right_speed):
        self.time_ns = time_ns
        self.left_speed = left_speed
        self.right_speed = right_speed


class Twist2DStamp:
    def __init__(self, time_ns, omega, v):
        self.time_ns = time_ns
        self.omega = omega
        self.v = v

