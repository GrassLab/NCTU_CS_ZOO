#!/usr/bin/env python3

from drivers.Adafruit_MotorHAT import Adafruit_MotorHAT
from math import fabs, floor


class DaguWheelsDriver:
    """Class handling communication with motors.
        Wraps the Adafruit API to talk to DC motors with a simpler interface.
        The class contains methods for creating PWM signals according to
        requested velocities. Also contains hardware addresses related to the
        motors.

        Args:
            debug (:obj:`bool`): If `True`, will print a debug message every time a PWM
               signal is sent.

    """

    LEFT_MOTOR_MIN_PWM = 60  #: Minimum speed for left motor
    LEFT_MOTOR_MAX_PWM = 255  #: Maximum speed for left motor
    RIGHT_MOTOR_MIN_PWM = 60  #: Minimum speed for right motor
    RIGHT_MOTOR_MAX_PWM = 255  #: Maximum speed for right motor
    SPEED_TOLERANCE = 1.e-2  #: Speed tolerance level

    def __init__(self, debug=False):

        self.motorhat = Adafruit_MotorHAT(addr=0x60)
        self.leftMotor = self.motorhat.getMotor(1)
        self.rightMotor = self.motorhat.getMotor(2)
        self.debug = debug

        self.leftSpeed = 0.0
        self.rightSpeed = 0.0
        self.updatePWM()

    def PWMvalue(self, v, minPWM, maxPWM):
        """Transforms the requested speed into an int8 number.
            Args:
                v (:obj:`float`): requested speed, should be between -1 and 1.
                minPWM (:obj:`int8`): minimum speed as int8
                maxPWM (:obj:`int8`): maximum speed as int8
        """
        pwm = 0
        if fabs(v) > self.SPEED_TOLERANCE:
            pwm = int(floor(fabs(v) * (maxPWM - minPWM) + minPWM))
        return min(pwm, maxPWM)

    def updatePWM(self):
        """Sends commands to the microcontroller.
            Updates the current PWM signals (left and right) according to the
            linear velocities of the motors. The requested speed gets
            tresholded.
        """
        vl = self.leftSpeed
        vr = self.rightSpeed

        pwml = self.PWMvalue(vl,
                             self.LEFT_MOTOR_MIN_PWM,
                             self.LEFT_MOTOR_MAX_PWM)
        pwmr = self.PWMvalue(vr,
                             self.RIGHT_MOTOR_MIN_PWM,
                             self.RIGHT_MOTOR_MAX_PWM)

        if fabs(vl) < self.SPEED_TOLERANCE:
            leftMotorMode = Adafruit_MotorHAT.RELEASE
            pwml = 0
        elif vl > 0:
            leftMotorMode = Adafruit_MotorHAT.FORWARD
        elif vl < 0:
            leftMotorMode = Adafruit_MotorHAT.BACKWARD

        if fabs(vr) < self.SPEED_TOLERANCE:
            rightMotorMode = Adafruit_MotorHAT.RELEASE
            pwmr = 0
        elif vr > 0:
            rightMotorMode = Adafruit_MotorHAT.FORWARD
        elif vr < 0:
            rightMotorMode = Adafruit_MotorHAT.BACKWARD

        self.leftMotor.setSpeed(pwml)
        self.leftMotor.run(leftMotorMode)
        self.rightMotor.setSpeed(pwmr)
        self.rightMotor.run(rightMotorMode)

    def setWheelsSpeed(self, left, right):
        """Sets speed of motors.

        Args:
           left (:obj:`float`): speed for the left wheel, should be between -1 and 1
           right (:obj:`float`): speed for the right wheel, should be between -1 and 1

        """

        self.leftSpeed = left
        self.rightSpeed = right
        self.updatePWM()

    def __del__(self):
        """Destructor method.
            Releases the motors and deletes tho object.
        """
        self.leftMotor.run(Adafruit_MotorHAT.RELEASE)
        self.rightMotor.run(Adafruit_MotorHAT.RELEASE)
        del self.motorhat


def test_wheels():
    import time
    driver = DaguWheelsDriver()
    driver.setWheelsSpeed(left=1, right=0)
    time.sleep(1)
    driver.setWheelsSpeed(left=0, right=0)
    time.sleep(0.5)
    driver.setWheelsSpeed(left=0, right=1)
    time.sleep(1)
    driver.setWheelsSpeed(left=0, right=0)
    time.sleep(0.5)
    driver.setWheelsSpeed(left=1, right=1)
    time.sleep(1)
    driver.setWheelsSpeed(left=0, right=0)
    time.sleep(0.5)
    driver.setWheelsSpeed(left=-1, right=-1)
    time.sleep(1)
    driver.setWheelsSpeed(left=0, right=0)
    time.sleep(0.5)
    pass


if __name__ == '__main__':
    test_wheels()
