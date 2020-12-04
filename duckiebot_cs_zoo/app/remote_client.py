import numpy as np
import os, cv2, joblib
import os.path as osp
from libs.tag_utils import get_detector, detect_apriltags, get_five_points
from remote_libs.car_control import CarControl
from remote_libs.light_control import LightControl
from remote_libs.camera_control import CameraControl


class DuckiebotClient:
    car: CarControl = None
    light: LightControl = None
    camera: CameraControl = None

    def __init__(self, server_port=54761):
        server_port = int(os.getenv('PORT', server_port))
        assert os.getenv('DUCKIE') is not None, \
            'Set Environment Variable DUCKIE to assign the hostname of your Duckiebot'
        self.car = CarControl(server_port)
        self.light = LightControl(server_port)
        self.camera = CameraControl(server_port)

        self.camera.restore_data(
            os.path.join(os.path.dirname(__file__), 'data', 'camera_calibration', 'camera_data.pkl'))

    def __del__(self):
        pass

    def set_velocity(self, omega, velocity,
                     show_pose=False):  # Omega is Rotation (car center) speed, Velocity is linear(forward) speed
        """
        :param omega: float, in range [-8, 8]
        :param velocity: float, in range [-1, 1]
        :return:
        """
        # Send to kinematic to calculate IK and then feed to motor and vel2pose
        self.car.move(velocity=velocity, omega=omega)

        if show_pose:
            pose = self.car.get_pose()
            t, x, y, theta = pose
            print('Current pose: x:{:.3f}, y:{:.3f}, theta:{:.3f} deg'.format(x, y,
                                                                              np.rad2deg(theta)))

    def get_rectified_image(self, img=None):  # Always use this for applications
        if img is None:
            img = self.camera.get_capture_img()
        return self.camera.get_rectified_image(img)

    def set_lights(self, light_dict):
        for pos, colors in light_dict.items():
            self.light.change_rgb(pos, red=colors[0], green=colors[1], blue=colors[2])

    def set_trim(self, trim):
        self.car.set_trim(trim)


def remote_control_demo():
    # Run on Desktop PC
    duckiebot = DuckiebotClient()
    duckiebot.set_lights({0: [0.2, 0.2, 0.2],
                          1: [0.2, 0.2, 0.2],
                          2: [0.2, 0.2, 0.2],
                          3: [0.2, 0.2, 0.2],
                          4: [0.2, 0.2, 0.2]})
    duckiebot.set_trim(0.01)  # From calibration
    while True:
        cmd = input('Cmd:')
        if cmd == 'sv':  # Set velocity
            in_data = input('Input omega,v:')
            omega, v = np.fromstring(in_data, dtype=float, sep=',')
            duckiebot.set_velocity(omega, v)
        elif cmd == 'kb':  # Keyboard Control
            while True:
                img = duckiebot.camera.get_stream_img()
                cv2.imshow("wsad_space_q", img)
                key = cv2.waitKey(30)
                if key != -1:
                    if chr(key) == 'w':
                        duckiebot.set_velocity(0, 0.5)
                    elif chr(key) == 's':
                        duckiebot.set_velocity(0, -0.5)
                    elif chr(key) == 'a':
                        duckiebot.set_velocity(2, 0)
                    elif chr(key) == 'd':
                        duckiebot.set_velocity(-2, 0)
                    elif chr(key) == 'q':
                        cv2.destroyAllWindows()
                        break
                    else:
                        duckiebot.set_velocity(0, 0)
        elif cmd == 'st':  # set trim
            trim = float(input('Input trim val:'))
            duckiebot.set_trim(trim)
        elif cmd == 'tags':
            detector = get_detector()
            tag_pts = get_five_points((0, 0), half=0.0125)
            while True:
                img = duckiebot.camera.get_rectified_image()
                pose_result, vis = detect_apriltags(img, tag_pts, duckiebot.camera.cam_mat, detector, return_vis=True)
                cv2.imshow('Tag Result', vis)
                key = cv2.waitKey(30)
                if key != -1:
                    if chr(key) == 'q':
                        break
        elif cmd == 'q':
            break


if __name__ == '__main__':
    remote_control_demo()
