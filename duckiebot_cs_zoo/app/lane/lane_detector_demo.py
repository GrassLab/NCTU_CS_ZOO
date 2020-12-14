import numpy as np
from app.remote_client import DuckiebotClient

import cv2
from libs.lane_detector import LaneDetector, Detection
import utils.image_processing as improc
from utils.image_processing import show_img

import warnings


def lane_detection_pipeline_single():
    """
    HW11-Step1

    In this example, you can tune and visualize our lane detector pipeline in each step.
    Please make sure you get good result in the final visualization in "improc.calc_normal"
    Note: Please create a clean workspace (better surrounded with white papers) for experiments

    In the following parts, typically
    - Press p to print the function parameters
    - Press q to leave the function

    Now, you need to run the pipeline first with all functions having vis=True flags (Use this at the first time)
    Then, fill-in all the parameters you desired (you should modify this file) and run the second time
        - Move the duckiebot to a different position for verification
        - Comment the lines with "Use this at the first time"
        - Uncomment the lines with "Use this later"

    Pay attention on the lines with "TODO: decide your xxx"
    After all parameters are tuned, please fill-in to the "detect_lane" function in your duckiebot_cs_zoo/libs/lane_detector.py
    """
    colors = ['green', 'blue', 'red', 'yellow', 'cyan']
    duckiebot = DuckiebotClient()
    # TODO: decide your light condition, if your room/space is not bright, you can turn on the lights, otherwise, "explicitly" turn off the lights
    duckiebot.set_lights(light_dict={0: [1, 1, 1],
                                     2: [1, 1, 1],
                                     4: [1, 1, 1]})
    lane_dt = LaneDetector(colors)
    COLOR_RANGE_FILE = 'my_colors.pkl'  # TODO: decide your filename
    """
    Step1: Color Balance Pipeline
    """
    print('Place the duckiebot in front of the colorful garage, press q to leave')
    while True:
        rect_img = duckiebot.get_rectified_image()
        k = show_img(rect_img, "Duckiebot", 30, destroy=False)
        if k == 'q':
            break
    percent = 30  # TODO: decide your parameters (percent, 20~50 are all fine depending on your environment)
    b1 = improc.color_balance(rect_img, percent=percent, clip_low=False, clip_high=True)
    b2 = improc.color_balance(rect_img, percent=percent, clip_low=True, clip_high=False)
    b3 = improc.color_balance(rect_img, percent=percent, clip_low=True, clip_high=True)
    # You will see four images
    # Original b1
    # b2       b3
    vis_p1 = np.vstack([np.hstack([rect_img, b1]), np.hstack([b2, b3])])
    show_img(vis_p1, 'ColorBalance')

    # Image to be processed
    img = b1  # TODO: decide your parameters (b1 or b2 or b3 and keep their clip_low/clip_high settings)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    """
    Step2: Color Segmentation Using HSV (Red "may" contains 2 ranges)
    """
    lane_dt.tune_color_ranges(img, file_name=COLOR_RANGE_FILE)  # Use this at the first time
    # lane_dt.load_color_ranges(COLOR_RANGE_FILE)  # Use this later
    """
    Step3: Find All Edges (binary map (either 0 or 255))
    """
    edges_all = improc.find_edges(img, vis=True)  # Use this at the first time
    # edges_all = improc.find_edges(img, canny_thresh1=80, canny_thresh2=200, canny_aperture_size=3)  # Use this later # TODO: decide your parameters (canny_thresh1/canny_thresh2/canny_aperture_size)

    # For each color range
    detection_result = {}
    for color, c_range in lane_dt.color_ranges.items():
        print(f'Processing color {color}')
        """
        Step4: Filter Edges  (HSV+Dilation)
        """
        in_range_map = c_range.hsv_in_range(hsv_img)
        # Dilation
        in_range_map = improc.dilation(in_range_map, vis=True)  # Use this at the first time
        # in_range_map = improc.dilation(in_range_map, cv2.MORPH_ELLIPSE, kernel_size=5)  # Use this later # TODO: decide your parameters (kernel_size)

        edges = cv2.bitwise_and(in_range_map, edges_all)  # edges for the current color (binary map)
        """
        Step5: Generate Hough Lines
        """
        lines = improc.hough_lines(edges, vis=True)  # Use this at the first time
        # lines = improc.hough_lines(edges, rho=1, theta_deg=1, threshold=1, minLineLength=2,
        #                            maxLineGap=1)  # Use this later # TODO: decide your parameters (threshold/minLineLength/maxLineGap)
        """
        Step6: Calculate Normal Vectors
        """
        if len(lines) > 0:
            # Keep the vis=True to check the final result before you use the detection result for car driving
            lines_new, normals, normal_types = improc.calc_normal(in_range_map, lines,
                                                                  vis=True)  # Use this for final check
            # lines_new, normals, normal_types = improc.calc_normal(in_range_map, lines, extend_l=5, vis=True) # TODO: decide your parameters (extend_l)
            if len(lines_new) > 0:
                detection_result[color] = Detection(lines_new, normals, in_range_map,
                                                    normal_types)  # Your detection_result for lane filter
            else:
                warnings.warn('No lines found for color {} after normal '.format(color))
                detection_result[color] = None
        else:
            warnings.warn('No lines found for color {}'.format(color))
            detection_result[color] = None


def main():
    # record_road_data()  # Video recording example, you "may" find this useful
    """
    HW11: Part1
    Understand each step by checking the code with "vis=True" for each step
    After you have good result, fill-in the default values to libs.lane_detector.detect_lane, which is almost identical to this function
    """
    lane_detection_pipeline_single()  # Example script, Make sure you (visually?) understand all the steps
    # fill-in the values to libs.lane_detector.detect_lane, we will use libs.lane_detector.detect_lane for the following parts


if __name__ == '__main__':
    main()
