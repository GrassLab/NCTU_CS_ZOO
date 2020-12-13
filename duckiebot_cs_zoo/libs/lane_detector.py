import os.path as osp, cv2
import joblib
from utils.fileutils import check_path
import utils.image_processing as improc
import warnings


class Detection:
    def __init__(self, lines, normals, in_range_map, normal_types):
        self.lines = lines
        self.normals = normals
        self.map = in_range_map
        self.normal_types = normal_types


class LaneDetector:
    DATA_PATH = check_path(osp.join(osp.dirname(osp.abspath(__file__)), 'data', 'lane_detector'))

    def __init__(self, color_names, color_range_file=None):
        self.color_names = color_names
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
        for color_name in self.color_names:
            print(f'Tuning ranges for {color_name}')
            self.color_ranges[color_name] = improc.RangeHSV.from_vis(img)  # Note: Red (may exist 2 valid ranges)
        if file_name is not None:
            joblib.dump({k: v.hsv_ranges for k, v in self.color_ranges.items()}, osp.join(self.DATA_PATH, file_name))
            print('Color ranges saved, filename={}'.format(file_name))

    def detect_lane(self, img, vis_normal=False):  # Main pipeline, TODO: Fill-in your parameters to each function
        """
        HW11-Step2

        For each function marked with TODO:Fill-in, please copy the settings in your duckiebot_cs_zoo/app/lane/lane_detector_demo.py
        YOU MUST CAREFULLY CHECK AGAIN BEFORE THE NEXT STAGE
        """
        # Step1: Color Balance Pipeline
        img = improc.color_balance(img, 30, True, True)  # TODO:Fill-in
        # Step2: Color Segmentation Using HSV
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Step3: Find All Edges and return a binary map (either 0 or 255)
        edges_all = improc.find_edges(img, 80, 250, 3)  # TODO:Fill-in

        # For each color range
        detection_result = {}
        for color, c_range in self.color_ranges.items():
            # Step4: Filter Edges
            in_range_map = c_range.hsv_in_range(hsv_img)
            # mask for current color
            in_range_map = improc.dilation(in_range_map, cv2.MORPH_ELLIPSE, 5)  # TODO:Fill-in
            # edges for the current color (binary map)
            edges = cv2.bitwise_and(in_range_map, edges_all)
            # Step5: Generate Hough Lines
            lines = improc.hough_lines(edges, rho=1, theta_deg=1, threshold=2, minLineLength=3,
                                       maxLineGap=1)  # TODO:Fill-in
            # Step6: Calculate Normal Vectors
            if len(lines) > 0:
                if vis_normal:
                    print(f'Visualizing normal for color "{color}"')
                lines_new, normals, normal_types = improc.calc_normal(in_range_map, lines, extend_l=5,
                                                                      vis=vis_normal)  # TODO:Fill-in (extend_l)
                if len(lines_new) > 0:
                    detection_result[color] = Detection(lines_new, normals, in_range_map, normal_types)
                else:
                    warnings.warn('No lines found for color {} after normal '.format(color))
                    detection_result[color] = None
            else:
                warnings.warn('No lines found for color {}'.format(color))
                detection_result[color] = None

        return detection_result
