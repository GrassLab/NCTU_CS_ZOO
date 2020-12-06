import numpy as np
import cv2

colors = {'red': (0, 0, 255),  # BGR
          'green': (0, 255, 0),
          'blue': (255, 0, 0),
          'black': (0, 0, 0),
          'yellow': (0, 255, 255)}


def get_detector():
    from pupil_apriltags import Detector
    return Detector(families='tag36h11',
                    nthreads=4,
                    quad_decimate=1.0,  # Full resolution
                    quad_sigma=0.0,
                    refine_edges=1,
                    decode_sharpening=0.25,
                    debug=0)


def get_five_points(center, half):
    center_x, center_y = center
    five_points = np.zeros((5, 3))
    five_points[0] = [center_x, center_y, 0]
    five_points[1] = [center_x - half, center_y - half, 0]
    five_points[2] = [center_x + half, center_y - half, 0]
    five_points[3] = [center_x + half, center_y + half, 0]
    five_points[4] = [center_x - half, center_y + half, 0]
    return five_points


def solve_pose(img_pts, obj_pts, cam_mat, flags=cv2.SOLVEPNP_IPPE):
    """
    :param img_pts: image points in 2D
    :param obj_pts: physical points in 3D
    :param cam_mat: camera matrix
    :return: camTobj: 4x4 np array
    """
    retval, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, cam_mat, np.zeros(4),
                                      flags=flags)  # cv2.SOLVEPNP_IPPE/SOLVEPNP_ITERATIVE
    rot_mat, _ = cv2.Rodrigues(rvec)
    camTobj = np.vstack((np.hstack((rot_mat, tvec.reshape(-1, 1))), np.array([0, 0, 0, 1])))
    return camTobj


def debug_tag(color, tag_result):
    for dt in tag_result:
        center = np.round(dt.center).astype(np.int)  # already in pixel coordinate(u,v)
        corners = np.round(dt.corners).astype(np.int)
        cv2.circle(color, (center[0], center[1]), 3, colors['black'], thickness=-1)
        cv2.circle(color, (corners[0][0], corners[0][1]), 3, colors['red'], thickness=-1)  # left_down
        cv2.circle(color, (corners[1][0], corners[1][1]), 3, colors['blue'], thickness=-1)  # right_down
        cv2.circle(color, (corners[2][0], corners[2][1]), 3, colors['yellow'], thickness=-1)  # right_up
        cv2.circle(color, (corners[3][0], corners[3][1]), 3, colors['green'], thickness=-1)  # left_up
    return color


def detect_apriltags(color_img, five_points, cam_mat, detector: Detector, return_vis=False):
    """
    :param color_img: cv2 image
    :param five_points: 5x3 np array (e.g., get_five_points((0,0),0.0125))
    :param cam_mat: camera matrix
    :param detector: detector from get_detector()
    :return:
    """
    grey_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    dt_result = detector.detect(grey_img)
    pose_result = {}
    for tag in dt_result:
        img_points = np.vstack((tag.center, tag.corners))
        pose = solve_pose(img_points, five_points, cam_mat)
        pose_result[tag.tag_id] = pose  # camTtag

    if return_vis:
        vis_color = debug_tag(color_img, dt_result)
        return pose_result, vis_color
    else:
        return pose_result


def get_pose_mapping(nx, ny, tsize, tspace):
    pose_mapping = {}
    """
    id: 5*3 array, center,left_down,.(CCW)..,left_up corners 
    """
    id_arrangement = np.arange(nx * ny).reshape(ny, nx).tolist()
    half = tsize / 2.0
    spacing = tsize * tspace

    for row, r_tags in enumerate(id_arrangement):
        for col, c_tag in enumerate(r_tags):
            center_x = (spacing + tsize) * col + half
            center_y = (spacing + tsize) * row + half
            pose_mapping[c_tag] = np.eye(4)
            pose_mapping[c_tag][0:3, 3] = [center_x, center_y, 0]
    return pose_mapping
