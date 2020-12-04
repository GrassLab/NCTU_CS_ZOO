import numpy as np, time
import os, cv2, math3d as m3d
from app.remote_client import DuckiebotClient
from libs.tag_utils import get_detector, detect_apriltags, get_five_points, get_pose_mapping


def get_current_2d_pose(tag_result, pose_mapping, carTcam):
    """
    :param tag_result: list of apriltag detections
    :param pose_mapping: Dictionary for 4x4 matrix of each tag in tag coordinate
    :param carTcam: Camera pose in the car coordinate
    :return:
        estimated_pos: numpy array of 2D position in the tag coordinate (unit:cm)
        rot: rotation (unit: degree)
    """

    """
    Your Code Here: Implement the coordinate transform and try to find the current position of your car in the tag coordinate 
    Hint:
        carTtag_id = carTcam @ camTtag_id
        oTcar = oTtag_id @ inv(carTtag_id)
    
    for tag_id, camTtag_id in tag_result.items():
        pass
    """
    # Return Value "Example" (Please modify fill them with your result)
    estimated_pos = np.array([0, 0]) * 100  # Meter to cm
    rot = np.rad2deg(0)  # Radian to degree
    return estimated_pos, rot


def hw10():
    duckiebot = DuckiebotClient()
    duckiebot.set_trim(0.01)  # From calibration
    detector = get_detector()
    tag_pts = get_five_points((0, 0), half=0.0125)
    pose_mapping = get_pose_mapping(nx=6, ny=9, tsize=0.025,
                                    tspace=0.25)  # Map each Tag to a 3D pose in the Tag coordinate

    while True:
        img = duckiebot.camera.get_rectified_image()
        tag_result, vis = detect_apriltags(img, tag_pts, duckiebot.camera.cam_mat, detector, return_vis=True)
        if len(tag_result) > 0:
            cur_pos, cur_rot = get_current_2d_pose(tag_result, pose_mapping, duckiebot.camera.data['cam_carTcam'])
            print(f'Car: Pos(cm): X={cur_pos[0]:.2f}, Y={cur_pos[1]:.2f}, Rot={cur_rot:.2f}')
        else:
            cur_rot, cur_pos = None, None
            print('No Tag Detected')
        """
        Your Code Here: Move your duckiebot to the right position by using cur_rot and cur_pos
        """
        cv2.imshow('Visualize Tags', vis)
        cv2.waitKey(10)


if __name__ == '__main__':
    hw10()
