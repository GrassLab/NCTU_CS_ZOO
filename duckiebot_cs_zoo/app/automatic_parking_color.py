import numpy as np
import cv2, time
from utils.image_processing import show_img
from libs.lane_filter_stop import LaneFilterStop
from libs.lane_filter_lr import LaneFilterLR
from libs.lane_detector import LaneDetector
from app.remote_client import DuckiebotClient


def HW11():
    colors = ['green', 'blue', 'red', 'yellow', 'cyan']
    duckiebot = DuckiebotClient()
    duckiebot.set_lights(light_dict={0: [1, 1, 1],
                                     2: [1, 1, 1],
                                     4: [1, 1, 1]})
    duckiebot.set_trim(0.01)  # From your calibration
    COLOR_RANGE_FILE = 'my_colors.pkl'  # TODO: Fill-in your filename
    H = duckiebot.camera.data['cam_H']
    lane_dt = LaneDetector(colors, COLOR_RANGE_FILE)
    lane_filter_stop = LaneFilterStop(H)
    lane_filter_lr = LaneFilterLR(H)
    frame = 0
    omega = vel = phi = 0
    t0 = t1 = time.time_ns()
    while True:
        frame += 1
        rect_img = duckiebot.get_rectified_image()  # You can get this from a video file XD
        k = show_img(rect_img, 'Rect', 10, False)
        if k == 'q':
            break

        detections = lane_dt.detect_lane(rect_img, vis_normal=False)
        lane_filter_lr.prediction(omega, vel, (time.time_ns() - t1) / 1e9)  # 0,0,0 is also ok
        lane_filter_stop.prediction(vel, phi, (time.time_ns() - t1) / 1e9)  # 0,0,0 is also ok

        # >>> No Detection Visualization
        # lane_filter_lr.update_posterior(detections, return_vis=False)
        # lane_filter_stop.update_posterior(detections, return_vis=False)
        # <<< No Detection visualization

        # >>> With Detection Visualization (for debug only), EXTREMELY SLOW
        v1 = lane_filter_lr.update_posterior(detections, return_vis=True)
        v2 = lane_filter_stop.update_posterior(detections, return_vis=True)

        k = show_img(np.hstack([rect_img, cv2.resize(v1, (rect_img.shape[1], rect_img.shape[0])),
                                cv2.resize(v2, (rect_img.shape[1], rect_img.shape[0]))]),
                     'lane_filter', 20, False)
        # <<< With Detection Visualization

        phi, d = lane_filter_lr.get_estimate()
        dist = lane_filter_stop.get_estimate()
        print(f'Estimate: D={d * 100:.2f}cm, Phi={np.rad2deg(phi):.2f},  DistStop={dist * 100:.2f}cm')
        if frame > 10:  # We ignore the first 10 (possibly in-accurate) frames
            """
            TODO: Design Your Move Logic
            """
            pass

    print(f'FPS={frame * 1e9 / (time.time_ns() - t0):.2f}')
    duckiebot.set_lights(light_dict={0: [0, 0, 0],
                                     2: [0, 0, 0],
                                     4: [0, 0, 0]})
    duckiebot.set_velocity(0, 0)


if __name__ == '__main__':
    HW11()
