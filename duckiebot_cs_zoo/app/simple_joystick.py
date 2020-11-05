try:
    from librarys.car_control import CarControl
except ImportError:
    from ..librarys.car_control import CarControl

duckie_car = CarControl(trim_val=0.0)

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
    elif key == "q":
        break
    else:
        continue

    duckie_car.move(
        velocity=velocity,
        omega=omega,
    )
