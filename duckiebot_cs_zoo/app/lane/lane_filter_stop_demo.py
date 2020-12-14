import numpy as np
import cv2
from utils.image_processing import show_img
from libs.lane_filter_stop import LaneFilterStop
from libs.lane_detector import LaneDetector
from app.remote_client import DuckiebotClient
import time


def lane_filter_STOP_demo():
    colors = ['green', 'blue', 'red', 'yellow', 'cyan']
    duckiebot = DuckiebotClient()
    # HW11-Step4 TODO: decide your light condition, if your room/space is not bright, you can turn on the lights, otherwise, "explicitly" turn off the lights
    duckiebot.set_lights(light_dict={0: [1, 1, 1],
                                     2: [1, 1, 1],
                                     4: [1, 1, 1]})
    H = duckiebot.camera.data['cam_H']
    COLOR_RANGE_FILE = 'my_colors.pkl'  # HW11-Step4 TODO: Fill-in your filename
    lane_dt = LaneDetector(colors, COLOR_RANGE_FILE)
    lane_filter_stop = LaneFilterStop(H)
    t1 = time.time()
    frame = 0
    while True:
        frame += 1
        rect_img = duckiebot.get_rectified_image()
        detections = lane_dt.detect_lane(rect_img, vis_normal=False)
        """
        Two steps for each new detection:
        1. lane_filter.prediction (optional, you can compare whether it's better)
        2. lane_filter.update_posterior
        """
        lane_filter_stop.prediction(v=0, phi=0, delta_t=0)  # phi from LaneFilterLR.get_estimate if available

        # >>> No Detection Visualization
        # lane_filter_stop.update_posterior(detections, return_vis=False)
        # k = show_img(rect_img, 'Rect', 10, False)  # With this line: ~10 fps, without: ~15fps
        # <<< No Detection visualization

        # >>> With Detection Visualization (for debug only), EXTREMELY SLOW
        vis = lane_filter_stop.update_posterior(detections, return_vis=True)  # Extremely slow, for debug only
        k = show_img(np.hstack([rect_img, cv2.resize(vis, (rect_img.shape[1], rect_img.shape[0]))]),
                     'LaneFilterStop', 20, False)
        # <<< With Detection Visualization

        dist = lane_filter_stop.get_estimate()
        print(f'Estimate: dist={dist * 100:.2f}cm')
        if k == 'q':
            break
    print(f'FPS={frame  / (time.time() - t1):.2f}')


if __name__ == '__main__':
    lane_filter_STOP_demo()
