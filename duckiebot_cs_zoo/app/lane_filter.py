import numpy as np
from app.lane_detector import LaneDetector
from scipy.ndimage.filters import gaussian_filter
import warnings
from functools import reduce
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import io, cv2

"""
LaneFilter: Lane Detector + Discrete Bayes Filter
Estimate the current pose of the duckiebot relative to the center of the lane
"""


def fig2img(fig, dpi=100):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class LandFilterParams:
    # All length represent in meters
    # Marker (tape) width
    marker_width = {'red': 0.05,
                    'white': 0.05,
                    'yellow': 0.025}
    # Lane width (the road in black)
    lane_width = 0.21

    # Detection region : region of line segments for consideration
    # Histogram setting
    # (phi,d) current planar pose relative to the lane center (+x forward, +y (align with d) on left)

    N_bins_phi = 31  # 16 or 31
    MAX_PHI_DEG = 75

    phi_bins = np.deg2rad(np.linspace(-MAX_PHI_DEG, MAX_PHI_DEG, N_bins_phi))
    delta_phi = phi_bins[1] - phi_bins[0]
    phi_min = phi_bins[0]

    N_bins_d = 31
    MAX_D_METER = 0.15

    d_bins = np.linspace(-MAX_D_METER, MAX_D_METER, N_bins_d)
    delta_d = d_bins[1] - d_bins[0]
    d_min = d_bins[0]

    # Init Distribution
    sigma_d_init = 0.01
    sigma_phi_init = 0.01
    # (lane_filter.belief*lane_filter.params.delta_phi*lane_filter.params.delta_d).sum()

    # Gaussian smooth kernel setting for prediction step
    use_gaussian_smooth = True
    sigma_d = 1.0
    sigma_phi = 2.0

    # Filter detected segments within a range in robot frame
    X_range = 0.35
    Y_range = 0.2

    # Filter detected normals within a reasonable angle range in robot frame
    angle_ignored = np.deg2rad(15)  # normal within 15 degrees to the x axis of the robot will be ignored

    # Color used
    color_used = ['yellow']  # ,'white']


class LaneFilter:
    def __init__(self, homography_mat):
        self.params = LandFilterParams
        self.H = homography_mat
        # np.meshgrid（values along each row, values along each column)
        self.d, self.phi = np.meshgrid(self.params.d_bins, self.params.phi_bins)  # Note: Shape is N_bins_phi x N_bins_d

        # Belief state (all states with probability)
        self.belief = np.zeros((self.params.N_bins_phi, self.params.N_bins_d))  # row_idx:phi, col_idx:d
        # self.belief[i,j]: probability of state p(phi==phi_bins[i],d==self.params.d_bins[j])
        self.init_belief()  # Multivariate Gaussian

    def init_belief(self):
        init_pos = np.zeros((*self.belief.shape, 2))
        cov_0 = [self.params.sigma_phi_init, self.params.sigma_d_init]  # Diagonal cov
        mean_0 = [0, 0]
        init_pos[:, :, 0] = self.phi
        init_pos[:, :, 1] = self.d
        self.belief = multivariate_normal.pdf(init_pos, mean=mean_0, cov=cov_0)
        # if you set sigma_phi_init and sigma_d_init with very small value (0.001)
        # (self.belief * self.params.delta_phi * self.params.delta_d).sum() ~=1

    def prediction(self, w, v, delta_t):  # P(x_t|u_t,x_{t-1})*bel(x_{t-1}) sum for all x_{t-1}
        """
        :param w: rotation speed of the duckiebot (since last estimation)
        :param v: translation speed of the duckiebot  (since last estimation)
        :param delta_t: delta time since last estimation
        :return: \overline{bel}
        """
        # From car dynamics, d and phi are 2d grid
        d_new = self.d + v * delta_t * np.sin(self.phi)  # Approximation
        phi_new = self.phi + w * delta_t

        # t1 = time.time_ns()
        """
        Step1: Calculate the new raw P(x_t|u_t,x_{t-1})
        """
        prob = np.zeros_like(self.belief)

        # Calculate new x_t for each x_{t-1} after the action (w,v) with time delta_t
        new_d_idx = ((d_new - self.params.d_min) // self.params.delta_d).astype(np.int)
        new_phi_idx = ((phi_new - self.params.phi_min) // self.params.delta_phi).astype(np.int)

        # Ignore the transition that are outside our scope
        valid_d = np.logical_and(new_d_idx < len(self.params.d_bins), new_d_idx >= 0)
        valid_phi = np.logical_and(new_phi_idx < len(self.params.phi_bins), new_phi_idx >= 0)

        valid = np.logical_and(np.logical_and(valid_d, valid_phi), self.belief > 0)

        # np.where: return two list, with each paired elements represents (row_idx_i,col_idx_i)
        valid_phi_2d, valid_d_2d = np.where(valid)  # Turn Boolean array into 2d indexing
        valid_d_1d = valid_d_2d.reshape(-1)
        valid_phi_1d = valid_phi_2d.reshape(-1)
        # Reshape and sum the new transition
        np.add.at(prob, (new_phi_idx[valid_phi_1d, valid_d_1d], new_d_idx[valid_phi_1d, valid_d_1d]),
                  self.belief[valid_phi_1d, valid_d_1d])

        # np.add.at is 3 times faster than the following equivalent for-loop, and is 10 times faster than the Equivalent Version

        # for phi_idx, d_idx in zip(valid_phi_1d, valid_d_1d):
        #     prob[new_phi_idx[phi_idx, d_idx], new_d_idx[phi_idx, d_idx]] += self.belief[phi_idx, d_idx]

        # print('T1={:.3f} us'.format((time.time_ns() - t1) / 1000))

        # """
        # Equivalent version
        # """
        # t2 = time.time_ns()
        # prob2 = np.zeros_like(self.belief)
        # for phi_idx in range(self.belief.shape[0]):
        #     for d_idx in range(self.belief.shape[1]):
        #         if self.belief[phi_idx, d_idx] > 0:
        #             phi_2 = int(
        #                 np.floor((phi_new[phi_idx, d_idx] - self.params.phi_min) / self.params.delta_phi))
        #             d_2 = int(
        #                 np.floor((d_new[phi_idx, d_idx] - self.params.d_min) / self.params.delta_d))
        #             if 0 <= phi_2 < self.params.N_bins_phi and 0 <= d_2 < self.params.N_bins_d:
        #                 prob2[phi_2, d_2] += self.belief[phi_idx, d_idx]
        # print('T2={:.3f} us'.format((time.time_ns() - t2) / 1000))
        # print(np.allclose(prob, prob2))

        """
        Step2: Gaussian Smoothing
        """
        if self.params.use_gaussian_smooth:
            prob = gaussian_filter(prob, [self.params.sigma_phi, self.params.sigma_d], mode='constant')
        if prob.sum() == 0:
            warnings.warn('Prediction step prob.sum()==0. Belief is not update.')
        else:
            self.belief = prob / prob.sum() / (
                    self.params.delta_phi * self.params.delta_d)  # Normalize, assume all probability are within the grid

    def update_posterior(self, detections, return_vis=False):
        vote_hist = np.zeros_like(self.belief)
        """
        Step1: Process sensor measurements by producing votes on phi and d for each line segments
        """
        votes_phi, votes_d = self.generate_votes(detections)
        """
        Step2: Generate sensor measurement probability
        """
        if len(votes_phi) > 0:
            # Select votes within the range of our bins # Similar to "Step1" in self.prediction
            new_d_idx = ((votes_d - self.params.d_min) // self.params.delta_d).astype(np.int)
            new_phi_idx = ((votes_phi - self.params.phi_min) // self.params.delta_phi).astype(np.int)

            # Ignore the votes that are outside our scope
            valid_d = np.logical_and(new_d_idx < len(self.params.d_bins), new_d_idx >= 0)
            valid_phi = np.logical_and(new_phi_idx < len(self.params.phi_bins), new_phi_idx >= 0)
            valid = np.logical_and(valid_d, valid_phi)

            # np.where: return two list, with each paired elements represents (row_idx_i,col_idx_i)
            valid_d_1d = new_d_idx[valid]
            valid_phi_1d = new_phi_idx[valid]
            # Reshape and do the voting by adding one to the possible state
            np.add.at(vote_hist, (valid_phi_1d, valid_d_1d), 1)
            if vote_hist.sum() > 0:
                vote_prob = vote_hist / vote_hist.sum()  # Normalize
                self.belief = np.multiply(self.belief, vote_prob)
                if self.belief.sum() == 0:
                    self.belief = vote_prob / (self.params.delta_phi * self.params.delta_d)
                else:
                    self.belief = self.belief / np.sum(self.belief)
            else:
                warnings.warn('No valid vote in update_posterior, belief was not updated')
        else:
            warnings.warn('No valid vote in update_posterior, belief was not updated')

        if return_vis:  # Slow !!
            if len(votes_d) > 0:
                votes_d = votes_d[valid]
                votes_phi = votes_phi[valid]
            fig = plt.figure(dpi=200)
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.hist(votes_d * 100, bins=100 * self.params.d_bins, label='d (cm)', histtype='bar',
                     color=[1.0, 0.2, 0.2],
                     edgecolor='black', linewidth=1.2)
            x_s, x_e = (self.params.d_bins[0] * 100).astype(np.int), (100 * self.params.d_bins[-1]).astype(np.int)
            step_x = max(1, (x_e - x_s) // 10)
            ax1.tick_params(axis='both', which='major', labelsize=7)
            ax1.set_xticks(np.arange(x_s, x_e, step_x))
            ax1.legend()
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.hist(np.rad2deg(votes_phi), bins=np.rad2deg(self.params.phi_bins), label='phi (deg)', histtype='bar',
                     color=[1.0, 1.0, 0.5], edgecolor='black', linewidth=1.2)

            x_s, x_e = np.rad2deg(self.params.phi_bins[0]).astype(np.int), np.rad2deg(self.params.phi_bins[-1]).astype(
                np.int)
            step_x = max(1, (x_e - x_s) // 10)
            ax2.tick_params(axis='both', which='major', labelsize=7)
            ax2.set_xticks(np.arange(x_s, x_e, step_x))
            ax2.legend()
            fig.tight_layout(rect=[0, 0.05, 1, 0.95])

            img = fig2img(fig, dpi=200)
            plt.close(fig)
            return img

    def get_normals(self, p1s, p2s, normal_types):
        t = p2s - p1s
        t = t / np.linalg.norm(t, axis=1).reshape(-1, 1)
        normals = np.hstack([-t[:, [1]], t[:, [0]]])  # Type1
        normals[normal_types == 2] *= -1  # Type2
        return normals

    def get_estimate(self):
        phi_idx, d_idx = np.unravel_index(self.belief.argmax(), self.belief.shape)
        phi_max = self.params.phi_min + (phi_idx + 0.5) * self.params.delta_phi
        d_max = self.params.d_min + (d_idx + 0.5) * self.params.delta_d
        return phi_max, d_max

    def generate_votes(self, detections):
        # Currently only consider yellow and white lane
        votes_phi = []
        votes_d = []

        for color, dt in detections.items():
            if color in self.params.color_used and dt is not None:
                p1s = self.to_robot_frame(dt.lines[:, :2])
                p2s = self.to_robot_frame(dt.lines[:, 2:])
                mids = (p1s + p2s) / 2.0
                normals = self.get_normals(p1s, p2s, dt.normal_types)
                valid_indices = self.filter_valid_range(mids)  # Use mid points
                if len(valid_indices) > 0:
                    mids = mids[valid_indices]
                    normals = normals[valid_indices]

                    angles = np.arctan2(normals[:, 1], normals[:, 0])  # 0~+-pi
                    """     +x
                            ^
                            |
                     case<- | case->
                    +y<-----|-----
                     case<- | case->
                            |
                    If a normal vec is heading toward the right side of +x (case ->) : the line belongs to the left  side of a color chunk
                                                      the left  side of +x (case <-) : the line belongs to the right side of a color chunk     
                    We ignore angles that are within angle_ignored to the x axis
                    """
                    # Determine the vector direction than check whether they are valid (using angle_ignored)

                    # case-> : angle in [-angle_ignored,-180+angle_ignored]
                    valid_left_select = np.logical_and(angles < -self.params.angle_ignored,
                                                       angles > (-np.pi + self.params.angle_ignored))
                    # case<-: angle in [angle_ignored,180-angle_ignored]
                    valid_right_select = np.logical_and(angles > self.params.angle_ignored,
                                                        angles < (np.pi - self.params.angle_ignored))

                    # print("Valid sum={}/{}".format(valid_left_select.sum() + valid_right_select.sum(),
                    #                                len(valid_right_select)))

                    # Calculate the votes according to their colors

                    if color == 'yellow':
                        if valid_left_select.sum() > 0:  # Process left YELLOW segments
                            mids_left = mids[valid_left_select]
                            normals_left = normals[valid_left_select]
                            d_left = -np.multiply(mids_left, normals_left).sum(axis=1) - self.params.marker_width[
                                'yellow']
                            d_robot = self.params.lane_width / 2.0 - d_left
                            phi_robot = -(np.arctan2(normals_left[:, 1], normals_left[:, 0]) + np.pi / 2.0)
                            votes_d.append(d_robot)
                            votes_phi.append(phi_robot)

                        if valid_right_select.sum() > 0:  # Process right YELLOW segments
                            mids_right = mids[valid_right_select]
                            normals_right = normals[valid_right_select]
                            d_right = np.multiply(mids_right, normals_right).sum(axis=1)
                            d_robot = self.params.lane_width / 2.0 - d_right
                            phi_robot = -(np.arctan2(normals_right[:, 1], normals_right[:, 0]) - np.pi / 2.0)
                            votes_d.append(d_robot)
                            votes_phi.append(phi_robot)

                    elif color == 'white':
                        if valid_left_select.sum() > 0:  # Process left WHITE segments
                            mids_left = mids[valid_left_select]
                            normals_left = normals[valid_left_select]
                            d_left = np.multiply(mids_left, normals_left).sum(axis=1)
                            d_robot = d_left - self.params.lane_width / 2.0
                            phi_robot = -(np.arctan2(normals_left[:, 1], normals_left[:, 0]) + np.pi / 2.0)
                            votes_d.append(d_robot)
                            votes_phi.append(phi_robot)

                        if valid_right_select.sum() > 0:  # Process right WHITE segments
                            mids_right = mids[valid_right_select]
                            normals_right = normals[valid_right_select]
                            d_right = -np.multiply(mids_right, normals_right).sum(axis=1) - self.params.marker_width[
                                'white']
                            d_robot = d_right - self.params.lane_width / 2.0
                            phi_robot = -(np.arctan2(normals_right[:, 1], normals_right[:, 0]) - np.pi / 2.0)
                            votes_d.append(d_robot)
                            votes_phi.append(phi_robot)
                else:
                    warnings.warn('No valid {} segments in robot frame'.format(color))
                    continue
        if len(votes_phi) > 0:
            return np.concatenate(votes_phi), np.concatenate(votes_d)
        else:
            return [], []

    def filter_valid_range(self, pts_robot):  # pts_robot in shape Nx2
        valid = reduce(np.logical_and, [pts_robot[:, 0] > 0,
                                        pts_robot[:, 0] < self.params.X_range,
                                        pts_robot[:, 1] > -self.params.Y_range,
                                        pts_robot[:, 1] < self.params.Y_range])
        return np.where(valid)[0]  # indices

    def to_robot_frame(self, pts_cam, normalize_vec=False):  # pts_cam in shape Nx2
        # From camera frame to robot frame
        points_cam = np.c_[pts_cam, np.ones(pts_cam.shape[0])].T  # 3xN
        points_robot = self.H @ points_cam  # 3x3 @ 3xN => 3xN
        points_robot = points_robot / points_robot[2, :]
        pts = points_robot[0:2, :].T  # Nx2
        if normalize_vec:
            pts = pts / np.linalg.norm(pts, axis=1).reshape(-1, 1)
        return pts


def lane_filter_test():
    from utils.video import VideoPlayer
    import os.path as osp, cv2
    import joblib, time
    app_video_path = osp.join(osp.dirname(osp.abspath(__file__)), 'data', 'videos')
    rect_path = osp.join(app_video_path, 'rect_1006_out.mov')
    rect_play = VideoPlayer(rect_path)
    client_data = joblib.load(
        osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'nodes', 'duckiebot_data', 'client_data.pkl'))
    H = client_data['homography']

    lane_dt = LaneDetector(color_range_file='color_ranges_1015.pkl')
    lane_filter = LaneFilter(H)
    while True:
        wait_time = int(1000 / rect_play.fps)  # ms

        rect_img = rect_play.read(loop=True)
        dt = lane_dt.detect_lane(rect_img)
        lane_filter.prediction(0, 0, 0.001)

        lane_filter.update_posterior(dt)
        cv2.imshow('Img', rect_img)

        # vis = lane_filter.update_posterior(dt, return_vis=True)
        # cv2.imshow('Img', np.hstack([rect_img, cv2.resize(vis, (rect_img.shape[1], rect_img.shape[0]))]))
        phi, d = lane_filter.get_estimate()
        print('Estimate: d={:.2f}cm, phi={:.2f} deg'.format(d * 100, np.rad2deg(phi)))
        k = cv2.waitKey(wait_time)
        if k != -1:
            if ord('q') == k:
                break
        # lane_filter.prediction()
        #

    pass


if __name__ == '__main__':
    # lane_filter_pipeline()
    lane_filter_test()
