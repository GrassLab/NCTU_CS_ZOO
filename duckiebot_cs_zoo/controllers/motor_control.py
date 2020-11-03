from drivers.dagu_wheels_driver import DaguWheelsDriver


class MotorControl(object):

    def __init__(self):
        self.motor_driver = DaguWheelsDriver()

    def move(self, motor_obj):
        self.motor_driver.setWheelsSpeed(motor_obj.left_speed, motor_obj.right_speed)
        print('Set motor speed: left={},right={}'.format(motor_obj.left_speed, motor_obj.right_speed))

