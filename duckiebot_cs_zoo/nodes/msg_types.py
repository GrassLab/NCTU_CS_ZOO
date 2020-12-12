"""
TODO: Define the messages more clearly for all communications, used for type-check
"""


class Pose2DStamp:
    def __init__(self, time_ns, x, y, theta):
        self.time_ns = time_ns
        self.x = x
        self.y = y
        self.theta = theta


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


class SetParamMsg:
    def __init__(self, param_dict: dict):
        self.param_dict = param_dict  # Key:Value


class CmdStr:  # Command string, TODO: Modify all related usages
    def __init__(self, cmd_str: str, cmd_val=None):
        self.cmd = cmd_str
        self.cmd_val = cmd_val


class FunctionCall:
    name: str = None
    args: list = None
    kwargs: dict = None

    def __init__(self, name: str, args: list = [], kwargs: dict = {}) -> None:
        self.name = name
        self.args = args
        self.kwargs = kwargs
