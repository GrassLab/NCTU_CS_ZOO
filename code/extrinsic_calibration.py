import numpy as np
import cv2
from camera_control import CameraControl

X_OFFSET = 0.191
Y_OFFSET = 0.093


def run_extrinsic_calibration(square_sz=0.031, board_size=(8, 6)):
    camera = CameraControl()
    print('Put the duckiebot on the calibration board')
    while True:
        img = camera.get_stream_img()
        rgb_rectified = camera.get_rectified_image(img)
        gray_rectified = cv2.cvtColor(rgb_rectified, cv2.COLOR_BGR2GRAY)

        corner_u, corner_v = board_size[0] - 1, board_size[1] - 1
        ret, corners = cv2.findChessboardCornersSB(gray_rectified, (corner_u, corner_v),
                                                   cv2.CALIB_CB_ACCURACY)
        # ret=True if Chessboard is found, then corners is with size (35,1,2)

        vis = cv2.drawChessboardCorners(rgb_rectified, (corner_u, corner_v), corners, ret)
        cv2.imshow("Chessboard Corner", vis)
        k = cv2.waitKey(30)
        if k == ord('q') and ret:
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

    # Later on, all data represented in image space (u,v) can be converted to robot coordinate (x,y) by using H
    # Under the important assumption that all content in the image are all at the ground plane (e.g., lanes/marker on the road)

    while True:
        img = camera.get_stream_img()
        rgb_rectified = camera.get_rectified_image(img)
        gray_rectified = cv2.cvtColor(rgb_rectified, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCornersSB(gray_rectified, (corner_u, corner_v),
                                                   cv2.CALIB_CB_ACCURACY)
        vis = cv2.drawChessboardCorners(rgb_rectified, (corner_u, corner_v), corners, ret)
        # Estimate the distance to the center corner
        if ret:
            if (corners[0])[0][0] < (corners[corner_u * corner_v - 1])[0][0] and \
                    (corners[0])[0][1] < (corners[corner_u * corner_v - 1])[0][1]:
                corners = corners[::-1, :, :]
            center_corner = corners[3, 0]
            vis = cv2.circle(vis, tuple(center_corner), 6, [0, 0, 255], -1)
            # Use ruler to measure the center_corner(red point) to the robot coordinate(center of the two wheels)
            # And compare the following result
            uv1 = np.r_[center_corner, 1].reshape((3, 1))
            xy1 = (H @ uv1).reshape(-1)
            xy1 /= xy1[2]
            # Initially, you should see similar result to the following=> Center corner: x=19.100, y=0.000, Dist=19.100 cm
            # You can now move the robot a little bit and compare the manually measured distance and the printed one (if the Chessboard is found)
            print('Center corner: x={:.3f}, y={:.3f}, Dist={:.3f} cm'.format(xy1[0] * 100, xy1[1] * 100,
                                                                             np.sqrt(xy1[0] ** 2 + xy1[1] ** 2) * 100))
        cv2.imshow('Vis', vis)
        k = cv2.waitKey(30)
        if k == ord('q'):
            break
        elif k == ord('s'):  # Press s to save and quit
            camera.set_extrinsic_calibration(H)
            print('Extrinsic calibration saved')
            break


if __name__ == '__main__':
    run_extrinsic_calibration()

