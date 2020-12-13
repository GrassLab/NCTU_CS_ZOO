import numpy as np, cv2
from utils.image_processing import show_img
from libs.lane_filter_lr import LaneFilterLR
from libs.lane_detector import LaneDetector
from app.remote_client import DuckiebotClient
import time


def lane_filter_LR_demo():
    colors = ['green', 'blue', 'red', 'yellow', 'cyan']
    duckiebot = DuckiebotClient()
    H = duckiebot.camera.data['cam_H']
    COLOR_RANGE_FILE = 'my_colors.pkl'  # HW11-Step3 TODO: Fill-in your filename
    lane_dt = LaneDetector(colors, COLOR_RANGE_FILE)
    lane_filter_lr = LaneFilterLR(H)
    t1 = time.time()
    frame = 0
    while True:
        frame += 1
        rect_img = duckiebot.get_rectified_image()
        detections = lane_dt.detect_lane(rect_img, vis_normal=False)  # vis_normal=True for normal check
        """
        Two steps for each new detection:
        1. lane_filter.prediction (optional, you can compare whether it's better)
        2. lane_filter.update_posterior
        """
        lane_filter_lr.prediction(omega=0, v=0, delta_t=0)

        # >>> No Detection Visualization
        # lane_filter_lr.update_posterior(detections, return_vis=False)
        # k = show_img(rect_img, 'Rect', 10, False)  # With this line: ~10 fps, without: ~15fps
        # <<< No Detection visualization

        # >>> With Detection Visualization (for debug only), EXTREMELY SLOW
        vis = lane_filter_lr.update_posterior(detections, return_vis=True)  # Extremely slow, for debug only
        k = show_img(np.hstack([rect_img, cv2.resize(vis, (rect_img.shape[1], rect_img.shape[0]))]),
                     'LaneFilterLR', 20, False)
        # <<< With Detection Visualization

        phi, d = lane_filter_lr.get_estimate()
        print('Estimate: d={:.2f}cm, phi={:.2f} deg'.format(d * 100, np.rad2deg(phi)))
        if k == 'q':
            break
    print(f'FPS={frame / (time.time() - t1):.2f}')


if __name__ == '__main__':
    lane_filter_LR_demo()
