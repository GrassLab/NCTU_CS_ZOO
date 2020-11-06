import numpy as np
import cv2, os, joblib
import os.path as osp
from utils.fileutils import check_path
from libs.camera_control import CameraControl
from datetime import datetime
import shutil


def collect_camera_calibration_chessboard(camera, square_sz, board_size, folder_path):
    """
    Please take around 20-30 images from varies angles, positions(left-right , far-near)
    :param camera: CameraControl object
    :param square_sz: Exact size of one square in the chessboard pattern (unit in meter)
    :param board_size: tuple (W,H), number of squares (black and white) for each row and column
    :param folder_path: image save folder path
    :return: intrinsic
    """
    check_path(folder_path)
    image_path = osp.join(folder_path, "image")
    check_path(image_path)
    # check if clean the previous images
    check_ret = input("Remove the previous images? (y/n)")
    if check_ret == "y":
        print("Removing all previous images...")
        for f in os.listdir(image_path):
            file_path = osp.join(image_path, f)
            try:
                if osp.isfile(file_path) or osp.islink(file_path):
                    os.unlink(file_path)
                elif osp.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    h, w = camera.get_stream_img().shape[:2]
    # Chessboard Mapping
    corner_u, corner_v = board_size[0] - 1, board_size[1] - 1
    print(
        'Collecting frames from camera,press 1 to collect, q to finish. Display shows once the Chessboard is detected')
    success_count = 0
    while True:
        rgb = camera.get_stream_img()
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCornersSB(gray, (corner_u, corner_v), flags=cv2.CALIB_CB_ACCURACY)
        vis = cv2.drawChessboardCorners(rgb.copy(), (corner_u, corner_v), corners, ret)
        vis = cv2.resize(vis, (320, 240))
        cv2.imshow("Chessboard Corner", vis)
        k = cv2.waitKey(30)
        if k == ord('1') and ret:
            cv2.imwrite(osp.join(image_path, f"{str(datetime.now())}.png"), rgb)
            success_count += 1
            print('Pattern {} saved'.format(success_count))
        elif k == ord('q'):
            break

def run_image_collect():
    DEFAULT_DATA_PATH = osp.join(osp.dirname(osp.abspath(__file__)), 'data', 'camera_calibration')
    camera = CameraControl()
    collect_camera_calibration_chessboard(camera, 0.031, (8, 6), DEFAULT_DATA_PATH)


if __name__ == '__main__':
    run_image_collect()
