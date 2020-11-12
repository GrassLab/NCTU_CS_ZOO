import serial
import time
try:
    from libs.car_control import CarControl
except ImportError:
    from ..libs.car_control import CarControl

# use CarControl library to control your car
# you can set up trim_val if your car cannot walk straightly when velocity is 1 and omega is 0
# -0.03 to 0.03 is the suitable range of trim_val, but it actually depends on the motors of your car 
duckie_car = CarControl(trim_val=0.0)

ser = serial.Serial('/dev/ttyUSB0', 9600)
time.sleep(1)
ser.flushInput()

while True:
    # clear the previous message in buffer
    ser.flushInput()
    # get the newest message from serial port
    data = ser.readline().decode("ascii")
    key = data[:-1]

    # omega range is -8 to 8. The negative number means turning right, and the positive number means turing left.
    # velocity range is 0 to 1. 0 is stop. 1 is moving forward with full speed. 0.3 is a good choice when you want to move forward.
    omega = 0
    velocity = 0
    if key == "move":
        omega = 0
        velocity = 0.3
    elif key == "left":
        omega = 2
        velocity = -2
    elif key == "right":
        omega = -2
        velocity = -2
    elif key == "stop":
        omega = 0
        velocity = 0
    else:
        omega = 0
        velocity = 0

    # call move with velocity and omega can control your little duck car
    duckie_car.move(
        velocity=velocity,
        omega=omega,
    )
