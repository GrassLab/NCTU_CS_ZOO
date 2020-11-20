import sys
import socket
import time
from enum import Enum
import builtins

import cv2
import numpy as np


class Command(Enum):
    STOP = 0
    FORWARD = 1
    BACKWARD = 2
    LEFT = 3
    RIGHT = 4


class BasicRemoteJoystick:
    motor_tcp: socket.socket = None
    motor_data = {
        Command.STOP: b'\x20',
        Command.FORWARD: b'\x77',
        Command.BACKWARD: b'\x73',
        Command.LEFT: b'\x61',
        Command.RIGHT: b'\x64',
    }
    light_udp: socket.socket = None
    light_addr: tuple = None
    light_data = {
        Command.STOP: b'\x64\x64\x64\x64\x64',
        Command.FORWARD: b'\x77\x77\x77\x64\x64',
        Command.BACKWARD: b'\x67\x67\x67\x72\x72',
        Command.LEFT: b'\x62\x64\x64\x62\x64',
        Command.RIGHT: b'\x64\x64\x62\x64\x62',
    }
    prev_cmd: Command = None

    def __init__(
            self,
            motor_tcp: socket.socket,
            light_udp: socket.socket,
            light_addr: tuple):
        self.motor_tcp = motor_tcp
        if self.motor_tcp is not None:
            self.motor_tcp.settimeout(1.0)
        self.light_udp = light_udp
        self.light_addr = light_addr

    def handle(self, cmd: Command) -> bool:
        # by pass if previous command is the same when STOP
        if cmd is Command.STOP and self.prev_cmd is cmd:
            return True
        if self.motor_tcp is not None:
            self.motor_tcp.send(self.motor_data[cmd])
        self.light_udp.sendto(self.light_data[cmd], self.light_addr)
        try:
            if self.motor_tcp is not None:
                ack: bytes = self.motor_tcp.recv(1)
            else:
                ack = b'\x21'
            ack = ack == b'\x21'
            if ack:
                self.prev_cmd = cmd
            else:
                print("!!Warnning!! no ack b'\x21' back")
        except socket.timeout:
            ack = False
            print("!!Warnning!! wait ack timeout")
            pass
        return ack


class CV2Terminal:
    window_name: str = None
    width = 640
    height = 480
    font_setting = dict(
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        thickness=2,
    )
    cur_y: int = 0
    foreground: np.ndarray = None
    background: np.ndarray = None

    def __init__(self, window_name: str) -> None:
        self.window_name = window_name
        # prepare terminal
        self.background = np.zeros((self.height, self.width, 3), np.uint8)
        self.background.fill(50)
        cv2.imshow(self.window_name, self.background)
        self.clear()

    def print(self, *values: object):
        s = " ".join(values)
        text_size, baseline = cv2.getTextSize(s, **self.font_setting)
        text_width, text_hegiht = text_size
        if self.cur_y + text_hegiht + baseline <= self.height:
            self.cur_y += text_hegiht + baseline
        else:
            self.clear()
            self.cur_y = text_hegiht + baseline
        cv2.putText(
            self.foreground, s, (baseline, self.cur_y),
            color=(255, 255, 255), **self.font_setting,
        )
        cv2.imshow(self.window_name, self.foreground)

    def clear(self):
        self.foreground = np.copy(self.background)


if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 3:
        print(f'Usage: {sys.argv[0]} host port')
        sys.exit(-1)
    # to prevent mDNS fail
    host: str = socket.gethostbyname(sys.argv[1])
    port: int = int(sys.argv[2])

    duckie_tcp = socket.socket(
        family=socket.AF_INET,
        type=socket.SOCK_STREAM,
    )
    duckie_udp = socket.socket(
        family=socket.AF_INET,
        type=socket.SOCK_DGRAM,
    )

    # wait until connect duckie
    deadline = time.time() + 1  # second
    print(f'connect to {host}:{port}')
    while duckie_tcp.connect_ex((host, port)) != 0:
        if time.time() > deadline:
            print('wait to timeout')
            res = input('Do you want to only test light? (Y/N) >>')
            if res == "N":
                sys.exit(-1)
            else:
                duckie_tcp = None
                break
        time.sleep(0.5)
        print(f'retry to connect {host}:{port}')

    remote_joystick = BasicRemoteJoystick(
        motor_tcp=duckie_tcp,
        light_udp=duckie_udp,
        light_addr=(host, port),
    )

    WINDOW_NAME = "Remote Joystick"
    term = CV2Terminal(WINDOW_NAME)

    # decorate print
    orig_print = builtins.print

    def power_print(*args, **kwargs):
        orig_print(*args, **kwargs)
        term.print(*args)
        return
    builtins.print = power_print

    print("pressed w, a, s, d and space as joystick when focus on Window")

    key2cmd = {
        ord('w'): Command.FORWARD,
        ord('a'): Command.LEFT,
        ord('s'): Command.BACKWARD,
        ord('d'): Command.RIGHT,
        ord(' '): Command.STOP,
        -1: Command.STOP,
    }
    while True:
        key: int = cv2.waitKey(500)

        if key == ord('q') or key == 27:
            break

        # key pressed
        if key in key2cmd:
            remote_joystick.handle(key2cmd[key])
            print(f'Handle {key2cmd[key]}')

    builtins.print = orig_print
    # to make sure duckie is stop
    remote_joystick.handle(Command.STOP)

    cv2.destroyAllWindows()
    if duckie_tcp is not None:
        duckie_tcp.close()
    duckie_udp.close()
    print('byebye')
