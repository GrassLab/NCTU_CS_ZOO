import numpy as np
import cv2
from os import path as osp
import joblib
from librarys.camera_control import CameraControl


def run_verify_calibration(square_sz=0.031, board_size=(8, 6)):
    camera = CameraControl()
    # set intrinsic data
    folder_path = osp.join(osp.dirname(osp.abspath(__file__)), 'data', 'camera_calibration')
    intrinsic_file_path = osp.join(folder_path, 'intrinsics.pkl')
    rms, camera_matrix, dist_coeffs = joblib.load(intrinsic_file_path)
    camera.set_cam_calibration(camera_matrix, dist_coeffs, 480, 640)
    # set extrinsic data
    H = joblib.load(osp.join(folder_path, "extrinsics.pkl"))

    while True:
        img = camera.get_stream_img()
        rgb_rectified = camera.get_rectified_image(img)
        # In cm
        H_inv = np.linalg.inv(H)
        ruler_x = np.arange(10, 50 + 1, 5) / 100  # cm to m
        ruler_y_left = np.full_like(ruler_x, 10) / 100
        ruler_y_right = -ruler_y_left
        n_pts_site = len(ruler_x)

        world_left = np.vstack([ruler_x, ruler_y_left, np.ones(n_pts_site)])
        world_right = np.vstack([ruler_x, ruler_y_right, np.ones(n_pts_site)])
        img_pts_left = (H_inv @ world_left)  # 3,N
        img_pts_right = (H_inv @ world_right)  # 3,N
        img_pts_left /= img_pts_left[[2], :]
        img_pts_right /= img_pts_right[[2], :]
        img_pts_left = np.round(img_pts_left).astype(np.int)
        img_pts_right = np.round(img_pts_right).astype(np.int)
        for i, x_dist in zip(range(n_pts_site), ruler_x):
            cv2.line(rgb_rectified, tuple(img_pts_left[0:2, i]), tuple(img_pts_right[0:2, i]), color=(0, 0, 255))
            cv2.putText(rgb_rectified, f'{int(x_dist * 100)}', (img_pts_left[0, i] - 35, img_pts_left[1, i]),
                        cv2.FONT_HERSHEY_PLAIN, 1, color=(0, 255, 255))

        cv2.imshow('Vis', rgb_rectified)
        k = cv2.waitKey(30)
        if k == ord('q'):
            break


if __name__ == '__main__':
    run_verify_calibration()

