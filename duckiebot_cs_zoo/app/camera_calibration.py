import numpy as np
import cv2, os, joblib
import os.path as osp
from utils.fileutils import check_path
from librarys.camera_control import CameraControl
from os import listdir
from os.path import isfile


def camera_calibration_chessboard(camera, square_sz, board_size, folder_path, intrinsics_name="intrinsics.pkl"):
    """
    Please take around 20-30 images from varies angles, positions(left-right , far-near)
    :param camera: CameraControl object
    :param square_sz: Exact size of one square in the chessboard pattern (unit in meter)
    :param board_size: tuple (W,H), number of squares (black and white) for each row and column
    :param folder_path: image and intrinsics.pkl save folder path
    :return: intrinsic
    """
    check_path(folder_path)
    image_path = osp.join(folder_path, "image")
    check_path(image_path)
    file_list = [f for f in listdir(image_path) if isfile(osp.join(image_path, f))]

    corner_u, corner_v = board_size[0] - 1, board_size[1] - 1
    objpoints = np.zeros((corner_u * corner_v, 3), dtype=np.float32)
    objpoints[:, 0:2] = np.mgrid[0:corner_u, 0:corner_v].T.reshape(-1, 2) * square_sz  # (su-1)x(sv-1)x3
    world_points = []
    img_points = []

    for f in file_list:
        if f[-3:] != "png" and f[-3:] != "jpg":
            continue
        rgb = cv2.imread(osp.join(image_path, f))
        h, w = rgb.shape[:2]
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCornersSB(gray, (corner_u, corner_v), flags=cv2.CALIB_CB_ACCURACY)
        if ret:
            # Ensure that all corner-arrays are going from top to bottom.
            # ref: https://github.com/ros-perception/image_pipeline/blob/noetic/camera_calibration/src/camera_calibration/calibrator.py#L214
            if corners[0, 0, 1] > corners[-1, 0, 1]:
                corners = np.copy(np.flipud(corners))
            world_points.append(objpoints)
            img_points.append(corners)

    if len(world_points) < 20:
        print('Please collect at least 20 images on the precious step!')
        return
    rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(world_points, img_points, (w, h),
                                                                None, None)
    print("Calibrated: rms={:5f}".format(rms))
    print('Intrinsic=\n{}\n Dist coeffs=\n{}'.format(camera_matrix, dist_coeffs))
    joblib.dump((rms, camera_matrix, dist_coeffs), osp.join(folder_path, intrinsics_name))
    # A good result should have RMS lower than 0.5
    if rms > 0.7:
        print('Please do the calibration again')


def validate_calibration(camera, square_sz, board_size, folder_path, intrinsics_name="intrinsics.pkl", alpha=0, draw_chessboard=False):
    """
    Validate (streaming the un-distort images)
    :param camera: CameraControl object
    :param square_sz: Exact size of one square in the chessboard pattern (unit in meter)
    :param board_size: tuple (W,H), number of squares for each row and column
    :param folder_path: intrinsics save folder_path
    :param alpha
    """
    _, camera_matrix, dist_coeffs = joblib.load(osp.join(folder_path, intrinsics_name))
    h, w = camera.get_stream_img().shape[:2]
    """
    # alpha=0: image is scaled and rectified so that all pixels are un-distroted (no black areas)
    # alpha=1: retain all pixels in the original image (with black areas)
    roi is valid un-distorted region for both cases
    """
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), alpha, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, newcameramtx, (w, h), 5)
    if roi == (0, 0, 0, 0):
        print("Fail to calibrate camera, return to the previous step...")
        return

    while True:
        img = camera.get_stream_img()
        dst = cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC)

        # # crop the image
        x, y, w, h = roi
        dst_crop = dst[y:y + h, x:x + w].copy()

        if draw_chessboard:
            corner_u, corner_v = board_size[0] - 1, board_size[1] - 1
            gray = cv2.cvtColor(dst_crop, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCornersSB(gray, (corner_u, corner_v))
            dst_crop = cv2.drawChessboardCorners(dst_crop, (corner_u, corner_v), corners, ret)
        cv2.imshow('Un-distort', dst)
        cv2.imshow('Rectified', dst_crop)
        k = cv2.waitKey(20)
        if ord('q') == k:
            break


def run_cam_calib():  # Camera calibration
    folder_path = osp.join(osp.dirname(osp.abspath(__file__)), 'data', 'camera_calibration')
    camera = CameraControl()
    camera_calibration_chessboard(camera, 0.031, (8, 6), folder_path)
    validate_calibration(camera, 0.031, (8, 6), folder_path, alpha=0, draw_chessboard=True) # compare alpha=0/1
    camera.stop_stream()


# Check rectified images
def check_rectified():
    camera = CameraControl()
    folder_path = osp.join(osp.dirname(osp.abspath(__file__)), 'data', 'camera_calibration')
    rms, camera_matrix, dist_coeffs = joblib.load(osp.join(folder_path, "intrinsics.pkl"))
    camera.set_cam_calibration(camera_matrix, dist_coeffs, 480, 640)

    while True:
        img = camera.get_stream_img()
        rect = camera.get_rectified_image(img)
        if rect.shape[0] == 0 or rect.shape[1] == 0:
            print("Fail to rectified images, please check if the previous step is successful")
            break

        cv2.imshow('Rect', rect)
        k = cv2.waitKey(20)
        if k != -1:
            if ord('q') == k:
                break
    camera.stop_stream()


if __name__ == '__main__':
    run_cam_calib()
    check_rectified()
