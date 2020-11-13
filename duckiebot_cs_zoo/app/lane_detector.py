import numpy as np
from nodes.duckiebot_client import DuckiebotClient

import os.path as osp, cv2
import joblib
from utils.video import VideoPlayer, VideoRecorder
from utils.fileutils import check_path
import utils.image_processing as improc
import warnings


class Detection:
    def __init__(self, lines, normals, in_range_map, normal_types):
        self.lines = lines
        # self.mid_points = (lines[:, :2] + lines[:, 2:]) / 2.0
        self.normals = normals
        self.map = in_range_map
        self.normal_types = normal_types


def show_img(img, title='Show Image', wait_key=0, destroy=True):
    cv2.imshow(title, img)
    k = cv2.waitKey(wait_key)
    if destroy:
        cv2.destroyWindow(title)
    return chr(k) if k > 0 else None


class LaneDetector:
    DATA_PATH = osp.join(osp.dirname(osp.abspath(__file__)), 'data', 'lane_detector')
    check_path(DATA_PATH)

    def __init__(self, color_range_file=None):
        self.color_ranges = {}
        if color_range_file is not None:
            self.load_color_ranges(color_range_file)

    def load_color_ranges(self, file_name):
        color_ranges = joblib.load(osp.join(self.DATA_PATH, file_name))
        self.color_ranges = {k: improc.RangeHSV(v) for k, v in color_ranges.items()}

    def tune_color_ranges(self, img, file_name=None):
        """
        :param img: image for color tuning
        :param file_name: if is not None, save the tuned color ranges in osp.join(self.DATA_PATH, file_name)
        :return:
        """

        print('Tuning ranges for Yellow')
        self.color_ranges['yellow'] = improc.RangeHSV.from_vis(img)
        print('Tuning ranges for White')
        self.color_ranges['white'] = improc.RangeHSV.from_vis(img)
        print('Tuning ranges for Red (may exist 2 valid ranges)')
        self.color_ranges['red'] = improc.RangeHSV.from_vis(img)  # TODO
        if file_name is not None:
            joblib.dump({k: v.hsv_ranges for k, v in self.color_ranges.items()}, osp.join(self.DATA_PATH, file_name))
            print('Color ranges saved, filename={}'.format(file_name))

    def detect_lane(self, img):  # Main pipeline # TODO: Fill-in your parameters to each function
        # Step1: Color Balance Pipeline
        img = improc.color_balance(img, percent=20, clip_low=True, clip_high=True)
        # Step2: Color Segmentation Using HSV
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Step3: Find All Edges
        edges_all = improc.find_edges(img, 80, 250, 3)  # binary map (either 0 or 255)

        # For each color range
        detection_result = {}
        for color, c_range in self.color_ranges.items():
            # Step4: Filter Edges
            in_range_map = c_range.hsv_in_range(hsv_img)
            in_range_map = improc.dilation(in_range_map, cv2.MORPH_ELLIPSE, 3)  # mask for current color
            edges = cv2.bitwise_and(in_range_map, edges_all)  # edges for the current color (binary map)
            # Step5: Generate Hough Lines
            lines = improc.hough_lines(edges, rho=1, theta_deg=1, threshold=2, minLineLength=3, maxLineGap=1)
            # Step6: Calculate Normal Vectors
            if len(lines) > 0:
                lines_new, normals, normal_types = improc.calc_normal(in_range_map, lines, extend_l=5)
                if len(lines_new) > 0:
                    detection_result[color] = Detection(lines_new, normals, in_range_map, normal_types)
                else:
                    warnings.warn('No lines found for color {} after normal '.format(color))
                    detection_result[color] = None
            else:
                warnings.warn('No lines found for color {}'.format(color))
                detection_result[color] = None

        return detection_result


def lane_detection_pipeline_single():
    # duckiebot = DuckiebotClient('duckiebotchai.local', 54761)

    app_video_path = osp.join(osp.dirname(osp.abspath(__file__)), 'data', 'videos')
    # raw_path = osp.join(app_video_path, 'raw_1006_out.mov')
    rect_path = osp.join(app_video_path, 'rect_1015_dt_2.mov')
    # raw_play = VideoPlayer(raw_path)
    rect_play = VideoPlayer(rect_path)
    lane_dt = LaneDetector()

    """
    Step1: Color Balance Pipeline
    """
    rect_img = rect_play.read(loop=True)
    # rect_img = duckiebot.get_rectified_image()
    b1 = improc.color_balance(rect_img, percent=20, clip_low=True, clip_high=False)
    b2 = improc.color_balance(rect_img, percent=20, clip_low=True, clip_high=True)
    vis_p1 = np.hstack([rect_img, b1, b2])
    show_img(vis_p1, 'ColorBalance')

    # Image to be processed
    img = b2
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    """
    Step2: Color Segmentation Using HSV
    """
    # lane_dt.tune_color_ranges(img, file_name='color_ranges_1015.pkl') # Red contains 2 ranges
    lane_dt.load_color_ranges('color_ranges_1015.pkl')
    """
    Step3: Find All Edges
    """
    # edges_all = improc.find_edges(b2, vis=True)
    edges_all = improc.find_edges(img, 80, 250, 3)  # binary map (either 0 or 255)

    # For each color range
    detection_result = {}
    for color, c_range in lane_dt.color_ranges.items():
        """
        Step4: Filter Edges 
        """
        in_range_map = c_range.hsv_in_range(hsv_img)
        # in_range_map = improc.dilation(map, vis=True)
        in_range_map = improc.dilation(in_range_map, cv2.MORPH_ELLIPSE, 3)  # mask for current color
        edges = cv2.bitwise_and(in_range_map, edges_all)  # edges for the current color (binary map)
        """
        Step5: Generate Hough Lines
        """
        # lines = improc.hough_lines(edges, vis=True)
        lines = improc.hough_lines(edges, rho=1, theta_deg=1, threshold=2, minLineLength=3, maxLineGap=1)
        """
        Step6: Calculate Normal Vectors
        """
        if len(lines) > 0:
            lines_new, normals, normal_types = improc.calc_normal(in_range_map, lines, vis=True)
            # lines_new, normals = improc.calc_normal(in_range_map, lines, extend_l=5)
            if len(lines_new) > 0:
                detection_result[color] = Detection(lines_new, normals, in_range_map, normal_types)
            else:
                warnings.warn('No lines found for color {} after normal '.format(color))
                detection_result[color] = None
        else:
            warnings.warn('No lines found for color {}'.format(color))
            detection_result[color] = None

    rect_play.release()


def record_road_data():
    app_video_path = osp.join(osp.dirname(osp.abspath(__file__)), 'data', 'videos')
    duckiebot = DuckiebotClient('duckiebotchai.local', 54761)
    raw_path = osp.join(app_video_path, 'raw_1015_dt_2.mov')  # _2 for manual color tuning
    rect_path = osp.join(app_video_path, 'rect_1015_dt_2.mov')
    raw_rec = VideoRecorder.from_img(duckiebot.get_stream_img(), raw_path)
    rect_rec = VideoRecorder.from_img(duckiebot.get_rectified_image(), rect_path)

    while True:
        raw_img = duckiebot.get_stream_img()
        rect_img = duckiebot.get_rectified_image(raw_img)
        cv2.imshow('raw', raw_img)
        cv2.imshow('rect', rect_img)
        raw_rec.write(raw_img)
        rect_rec.write(rect_img)
        k = cv2.waitKey(10)
        if k == ord('q'):
            break
    raw_rec.release()
    rect_rec.release()
    # Replay video

    raw_play = VideoPlayer(raw_path)
    rect_play = VideoPlayer(rect_path)
    # Info
    print('Video: Len={}, FPS={}'.format(len(raw_play), raw_play.fps))
    while True:
        wait_time = int(1000 / raw_play.fps)
        raw_img = raw_play.read(loop=True)
        rect_img = rect_play.read(loop=True)
        cv2.imshow('raw', raw_img)
        cv2.imshow('rect', rect_img)
        k = cv2.waitKey(wait_time)
        if k == ord('q'):
            break
    raw_play.release()
    rect_play.release()


def main():
    # record_road_data()
    lane_detection_pipeline_single()  # Example script


if __name__ == '__main__':
    main()
