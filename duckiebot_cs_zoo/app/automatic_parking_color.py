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
    # HW11-Step6 TODO: decide your light condition, if your room/space is not bright, you can turn on the lights, otherwise, "explicitly" turn off the lights
    duckiebot.set_lights(light_dict={0: [1, 1, 1],
                                     2: [1, 1, 1],
                                     4: [1, 1, 1]})
    duckiebot.set_trim(0.01)  # From your calibration
    COLOR_RANGE_FILE = 'my_colors.pkl'  # HW11-Step6 TODO: Fill-in your filename
    H = duckiebot.camera.data['cam_H']
    lane_dt = LaneDetector(colors, COLOR_RANGE_FILE)
    lane_filter_stop = LaneFilterStop(H)
    lane_filter_lr = LaneFilterLR(H)
    frame = 0
    omega = vel = phi = 0
    t0 = t1 = time.time()
    while True:
        frame += 1
        rect_img = duckiebot.get_rectified_image()

        detections = lane_dt.detect_lane(rect_img, vis_normal=False)
        lane_filter_lr.prediction(omega, vel, (time.time() - t1))  # 0,0,0 is also ok
        lane_filter_stop.prediction(vel, phi, (time.time() - t1))  # 0,0,0 is also ok

        # >>> No Detection Visualization
        # k = show_img(rect_img, 'Rect', 10, False)
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
        if k == 'q':
            break
        if frame > 10:  # We ignore the first 10 (possibly in-accurate) frames
            """
            HW11-Step6 TODO: Design Your Move Logic
            """
            pass

    print(f'FPS={frame / (time.time() - t0):.2f}')
    duckiebot.set_lights(light_dict={0: [0, 0, 0],
                                     2: [0, 0, 0],
                                     4: [0, 0, 0]})
    duckiebot.set_velocity(0, 0)


if __name__ == '__main__':
    HW11()
