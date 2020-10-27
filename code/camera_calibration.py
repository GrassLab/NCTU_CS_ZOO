import numpy as np
import cv2, os, joblib
import os.path as osp
from utils.fileutils import check_path
from camera_control import CameraControl


def camera_calibration_chessboard(camera, square_sz, board_size, file_path):
    """
    Please take around 20-30 images from varies angles, positions(left-right , far-near)
    :param camera: CameraControl object
    :param square_sz: Exact size of one square in the chessboard pattern (unit in meter)
    :param board_size: tuple (W,H), number of squares (black and white) for each row and column
    :param file_path: intrinsics save file path
    :return: intrinsic
    """
    # os.putenv('DISPLAY', ':0')
    check_path(osp.dirname(file_path))
    h, w = camera.get_stream_img().shape[:2]
    # Chessboard Mapping
    corner_u, corner_v = board_size[0] - 1, board_size[1] - 1
    objpoints = np.zeros((corner_u * corner_v, 3), dtype=np.float32)
    objpoints[:, 0:2] = np.mgrid[0:corner_u, 0:corner_v].T.reshape(-1, 2) * square_sz  # (su-1)x(sv-1)x3
    world_points = []
    img_points = []
    print(
        'Collecting frames from camera,press 1 to collect, q to finish. Display shows once the Chessboard is detected')
    while True:
        rgb = camera.get_stream_img()
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCornersSB(gray, (corner_u, corner_v), flags=cv2.CALIB_CB_ACCURACY)
        vis = cv2.drawChessboardCorners(rgb, (corner_u, corner_v), corners, ret)
        cv2.imshow("Chessboard Corner", vis)
        k = cv2.waitKey(30)
        if k == ord('1') and ret:
            # Ensure that all corner-arrays are going from top to bottom.
            # ref: https://github.com/ros-perception/image_pipeline/blob/noetic/camera_calibration/src/camera_calibration/calibrator.py#L214
            if corners[0, 0, 1] > corners[-1, 0, 1]:
                corners = np.copy(np.flipud(corners))
                print("flip!")
            world_points.append(objpoints)
            world_points.append(objpoints)
            img_points.append(corners)
            print('Pattern {} saved'.format(len(world_points)))
        elif k == ord('q'):
            if len(world_points) < 20:
                print('Please collect at least 20 images')
            else:
                break

    rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(world_points, img_points, (w, h),
                                                                None, None)
    print("Calibrated: rms={:5f}".format(rms))
    print('Intrinsic=\n{}\n Dist coeffs=\n{}'.format(camera_matrix, dist_coeffs))
    joblib.dump((rms, camera_matrix, dist_coeffs), file_path)
    # A good result should have RMS lower than 0.5
    if rms > 0.7:
        print('Please do the calibration again')


def validate_calibration(camera, square_sz, board_size, file_path, alpha=0, draw_chessboard=False):
    """
    Validate (streaming the un-distort images)
    :param camera: CameraControl object
    :param square_sz: Exact size of one square in the chessboard pattern (unit in meter)
    :param board_size: tuple (W,H), number of squares for each row and column
    :param file_path: intrinsics save file_path
    :param alpha
    """
    _, camera_matrix, dist_coeffs = joblib.load(file_path)
    h, w = camera.get_stream_img().shape[:2]
    """
    # alpha=0: image is scaled and rectified so that all pixels are un-distroted (no black areas)
    # alpha=1: retain all pixels in the original image (with black areas)
    roi is valid un-distorted region for both cases
    """
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), alpha, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, newcameramtx, (w, h), 5)

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
    DEFAULT_DATA_PATH = osp.join(osp.dirname(osp.abspath(__file__)), 'data', 'camera_calibration')
    file_path = osp.join(DEFAULT_DATA_PATH, 'intrinsics.pkl')
    camera = CameraControl()
    camera_calibration_chessboard(camera, 0.031, (8, 6), file_path)
    validate_calibration(camera, 0.031, (8, 6), file_path, alpha=0, draw_chessboard=True) # compare alpha=0/1

    # save the result
    rms, camera_matrix, dist_coeffs = joblib.load(file_path)
    camera.set_cam_calibration(camera_matrix, dist_coeffs, 480, 640)


# Check rectified images
def check_rectified():
    while True:
        img = camera.get_stream_img()
        rect = camera.get_rectified_image(img)
        cv2.imshow('Rect', rect)
        k = cv2.waitKey(20)
        if k != -1:
            if ord('q') == k:
                break


if __name__ == '__main__':
    run_cam_calib()
    check_rectified()
