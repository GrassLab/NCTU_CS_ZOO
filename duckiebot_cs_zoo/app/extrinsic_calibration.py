import numpy as np
import cv2
from os import path as osp
import joblib
from librarys.camera_control import CameraControl

X_OFFSET = 0.191
Y_OFFSET = 0.093


def run_extrinsic_calibration(square_sz=0.031, board_size=(8, 6)):
    camera = CameraControl()
    # set intrinsic data
    folder_path = osp.join(osp.dirname(osp.abspath(__file__)), 'data', 'camera_calibration')
    intrinsic_file_path = osp.join(folder_path, 'intrinsics.pkl')
    rms, camera_matrix, dist_coeffs = joblib.load(intrinsic_file_path)
    camera.set_cam_calibration(camera_matrix, dist_coeffs, 480, 640)
    # extrinsic calibration
    print('Put the duckiebot on the calibration board')
    while True:
        img = camera.get_stream_img()
        rgb_rectified = camera.get_rectified_image(img)
        gray_rectified = cv2.cvtColor(rgb_rectified, cv2.COLOR_BGR2GRAY)

        corner_u, corner_v = board_size[0] - 1, board_size[1] - 1
        ret, corners = cv2.findChessboardCorners(gray_rectified, (corner_u, corner_v),
                                                 cv2.CALIB_CB_ACCURACY)
        # ret=True if Chessboard is found, then corners is with size (35,1,2)

        vis = cv2.drawChessboardCorners(rgb_rectified, (corner_u, corner_v), corners, ret)
        cv2.imshow("Chessboard Corner", vis)
        k = cv2.waitKey(30)
        if k == ord(' ') and ret:
            break
    # Check board points

    # Build world points
    src_pts = []
    origin_offset = np.array([X_OFFSET, -Y_OFFSET])  # origin_offset at bottom right corner
    for r in range(corner_v):
        for c in range(corner_u):
            # Bottom right -> Upper left
            src_pts.append(
                np.array([r * square_sz, c * square_sz], dtype='float32') + origin_offset)

    # OpenCV labels corners left-to-right, top-to-bottom
    # Reverse order if first corner point is the top-left corner (bottom-right is the desired origin)
    if (corners[0])[0][0] < (corners[corner_u * corner_v - 1])[0][0] and \
            (corners[0])[0][1] < (corners[corner_u * corner_v - 1])[0][1]:
        corners = corners[::-1, :, :]
    H, status = cv2.findHomography(corners.reshape(len(corners), 2), np.vstack(src_pts), cv2.RANSAC)
    print('H={}'.format(H))

    folder_path = osp.join(osp.dirname(osp.abspath(__file__)), 'data', 'camera_calibration')
    joblib.dump(H, osp.join(folder_path, "extrinsics.pkl"))


if __name__ == '__main__':
    run_extrinsic_calibration()

