import numpy as np
import cv2, time
from utils.image_processing import show_img
from libs.lane_filter_stop_red import LaneFilterStop  # Note: new module, not your HW file
from libs.lane_filter_lr_duckie import LaneFilterLRduckie  # Note: new module, not your HW file
from libs.lane_detector import LaneDetector
from app.remote_client import DuckiebotClient


def HW11():
    colors = ['red', 'yellow', 'white']
    duckiebot = DuckiebotClient()
    # HW11-Step6 TODO: decide your light condition, if your room/space is not bright, you can turn on the lights, otherwise, "explicitly" turn off the lights
    duckiebot.set_lights(light_dict={0: [0, 0, 0],
                                     2: [0, 0, 0],
                                     4: [0, 0, 0]})
    duckiebot.set_trim(0.01)  # From your calibration
    COLOR_RANGE_FILE = 'my_colors_challenge11.pkl'  # HW11-Step6 TODO: Fill-in your filename for challenge11
    COLOR_BALANCE_FILE = 'my_balance_challenge11.pkl'  # HW11-Step6 TODO: Fill-in your filename challenge11
    H = duckiebot.camera.data['cam_H']
    lane_dt = LaneDetector(colors, COLOR_RANGE_FILE, COLOR_BALANCE_FILE)
    lane_filter_stop = LaneFilterStop(H)
    lane_filter_lr = LaneFilterLRduckie(H)
    frame = 0
    omega = vel = phi = 0
    t0 = t1 = time.time()

    while True:
        frame += 1
        rect_img = duckiebot.get_rectified_image()

        detections = lane_dt.detect_lane(rect_img, vis_normal=False)
        lane_filter_lr.prediction(omega, vel, (time.time() - t1))  # 0,0,0 is also ok
        lane_filter_stop.prediction(vel, phi, (time.time() - t1))  # 0,0,0 is also ok
        t1 = time.time()
        # >>> No Detection Visualization
        k = show_img(rect_img, 'Rect', 10, False)
        lane_filter_lr.update_posterior(detections, return_vis=False)
        lane_filter_stop.update_posterior(detections, return_vis=False)
        # <<< No Detection visualization

        # >>> With Detection Visualization (for debug only), EXTREMELY SLOW
        # v1 = lane_filter_lr.update_posterior(detections, return_vis=True)
        # v2 = lane_filter_stop.update_posterior(detections, return_vis=True)
        #
        # k = show_img(np.hstack([rect_img, cv2.resize(v1, (rect_img.shape[1], rect_img.shape[0])),
        #                         cv2.resize(v2, (rect_img.shape[1], rect_img.shape[0]))]),
        #              'lane_filter', 20, False)
        # <<< With Detection Visualization

        detected_red_segments = 0 if detections['red'] is None else len(detections['red'].lines)
        # Challenge11: You should consider the number of detected red segments when you decide whether to use the result, below is an example
        print(f'Red Detected N={detected_red_segments}')
        phi, d = lane_filter_lr.get_estimate()
        dist = lane_filter_stop.get_estimate()
        """ 
        # TA's Example
        if detected_red_segments <10:  # Example
            dist = -1  # Example
        """
        print(f'Estimate: D={d * 100:.2f}cm, Phi={np.rad2deg(phi):.2f},  DistStop={dist * 100:.2f}cm')
        if k == 'q':
            break
        if frame > 10:  # We ignore the first 10 (possibly in-accurate) frames
            """
            HW11-Step6 TODO: Design Your Move Logic
            """

            """
            # TA's Example Control Logic (May need some tuning)
            
            d_cm = d * 100
            angle = np.rad2deg(phi)
            error_ang = 0 - angle
            error_d_cm = 0 - d_cm
            # Constants
            P_angle_div = 0.5  # 0.5
            P_d_div = 0.5  # 0.5
            OMEGA_D_MAX = 2  # 2
            OMEGA_ROT_MAX = 2  # 2
            D_OMIT_CM = 1.5  # 1.5
            ANGLE_OMIT_DEG = 3  # 5
            vel=0.15

            omega_rot_P = np.clip(error_ang / P_angle_div, -OMEGA_ROT_MAX, OMEGA_ROT_MAX)
            omega_d_P = np.clip(error_d_cm / P_d_div, -OMEGA_D_MAX, OMEGA_D_MAX)

            omega_rot_P = 0 if np.abs(error_ang) < ANGLE_OMIT_DEG else omega_rot_P
            omega_d_P = 0 if np.abs(error_d_cm) < D_OMIT_CM else omega_d_P
            omega = omega_rot_P + omega_d_P
            duckiebot.set_velocity(omega,vel)
            """

            # TODO: Determine when you should stop your car
            # if np.abs(dist) < 0.01: # Example
            #     pass  # You should stop your car

    print(f'FPS={frame / (time.time() - t0):.2f}')
    duckiebot.set_lights(light_dict={0: [0, 0, 0],
                                     2: [0, 0, 0],
                                     4: [0, 0, 0]})
    duckiebot.set_velocity(0, 0)


if __name__ == '__main__':
    HW11()
