import numpy as np
from scipy.ndimage.filters import gaussian_filter
import warnings
from functools import reduce
import matplotlib.pyplot as plt
from utils.image_processing import fig2img

"""
LaneFilterStop: Stop Position Lane Filter
Estimate the current distance to the stop position using horizontal lines

---  RED  --- +16.6cm

--- YELLOW--- + 7.6cm
--- STOP+ --- 0cm

---  Cyan --- -8.95cm


"""


class LaneFilterParamsStop:
    marker_width = {'red': 0.02,  # Far
                    'yellow': 0.02,
                    'cyan': 0.02}  # Close
    # Color chunk (bottom edge) distance to the center point for parking
    line_dist = {
        'red': 0.166,
        'yellow': 0.076,
        'cyan': -0.0895
    }

    N_bins_dist = 71
    MIN_D_METER = -0.25
    MAX_D_METER = 0.25

    dist_bins = np.linspace(MIN_D_METER, MAX_D_METER, N_bins_dist)
    delta_dist = dist_bins[1] - dist_bins[0]
    dist_min = dist_bins[0]

    # Init Distribution
    sigma_dist_init = 0.01

    # Gaussian smooth kernel setting for prediction step
    use_gaussian_smooth = True
    sigma_dist = 0.5

    # Filter detected segments within a range in robot frame
    X_range = 0.50  # +
    Y_range = 0.1  # +-

    # Color used
    color_used = ['yellow', 'red', 'cyan']
    angle_considered = np.deg2rad(45)  # angle to +-x axis

    # Moving average output filter
    use_moving_avg = True
    mov_avg_window_size = 4
    mov_avg_policy = 'mean'  # mean or median


class LaneFilterStop:
    def __init__(self, homography_mat):
        self.params = LaneFilterParamsStop()
        self.H = homography_mat
        self.dist = self.params.dist_bins

        # Belief state (all states with probability)
        self.belief = np.full_like(self.dist, 1 / len(self.dist))
        self.out_pred_buff = np.zeros(self.params.mov_avg_window_size)

    def prediction(self, v, phi, delta_t):
        dist_new = self.dist - v * delta_t * np.cos(phi)
        prob = np.zeros_like(self.belief)
        new_dist_idx = ((dist_new - self.params.dist_min) // self.params.delta_dist).astype(np.int)
        valid = np.logical_and(new_dist_idx < len(self.params.dist_bins), new_dist_idx >= 0)
        np.add.at(prob, new_dist_idx[valid], self.belief[valid])  # Transition
        """
        Step2: Gaussian Smoothing
        """
        if self.params.use_gaussian_smooth:
            prob = gaussian_filter(prob, [self.params.sigma_dist], mode='constant')
        if prob.sum() == 0:
            warnings.warn('Prediction step prob.sum()==0. Belief is not update.')
        else:
            self.belief = prob / prob.sum()

    def update_posterior(self, detections, return_vis=False):
        vote_hist = np.zeros_like(self.belief)
        votes_dist = self.generate_votes(detections)

        if len(votes_dist) > 0:
            new_dist_idx = ((votes_dist - self.params.dist_min) / self.params.delta_dist).astype(np.int)
            valid_dist = np.logical_and(new_dist_idx < len(self.params.dist_bins), new_dist_idx >= 0)
            np.add.at(vote_hist, new_dist_idx[valid_dist], 1)
            if vote_hist.sum() > 0:
                vote_prob = vote_hist / vote_hist.sum()  # Normalize
                if self.params.use_gaussian_smooth:
                    vote_prob = gaussian_filter(vote_prob, [self.params.sigma_dist], mode='constant')
                self.belief = np.multiply(self.belief, vote_prob)
                if self.belief.sum() == 0:  # only use sensor data to determine the new belief
                    self.belief = vote_prob
                else:
                    self.belief = self.belief / np.sum(self.belief)
            else:
                warnings.warn('No valid vote in update_posterior, belie9f was not updated')
        else:
            warnings.warn('No valid vote in update_posterior, belief was not updated')

        if return_vis:  # Extremely Slow !!
            if len(votes_dist) > 0:
                votes_dist = votes_dist[valid_dist]
            fig = plt.figure(dpi=200)
            ax1 = fig.add_subplot(1, 1, 1)
            ax1.hist(votes_dist * 100, bins=100 * self.params.dist_bins, label='dist (cm)', histtype='bar',
                     color=[1.0, 0.2, 0.2],
                     edgecolor='black', linewidth=1.2)
            x_s, x_e = (self.params.dist_bins[0] * 100).astype(np.int), (100 * self.params.dist_bins[-1]).astype(np.int)
            step_x = max(1, (x_e - x_s) // 10)
            ax1.tick_params(axis='both', which='major', labelsize=7)
            ax1.set_xticks(np.arange(x_s, x_e, step_x))
            ax1.legend()
            fig.tight_layout(rect=[0, 0.05, 1, 0.95])
            img = fig2img(fig, dpi=200)
            plt.close(fig)
            del fig
            return img

    def generate_votes(self, detections):
        # Currently only consider yellow and red lanes
        votes_dist = []
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
                    """ 
                    All vectors are in the duckiebot frame
                    **A normal vector is heading directly to it's corresponding color chunk.**
                    
                    Duckiebot frame
                            +x
                            ^
                            |
                            |
                    +y<-----|-----
                            | 
                            |
                    If a normal vec is heading approximately to 1. the direction of -x (case ->) : this line belongs to the top side(further to duckiebot) of a color chunk
                                                                2. the direction of +x (case <-) : this line belongs to the bottom side(closer to duckiebot) of a color chunk     
                    We ignore angles that are "outside" angle_considered to the x axis to reduce ambiguity
                    """

                    # Determine the vector direction then check whether they are valid
                    # normal starts from the top side
                    valid_top_select = (np.pi - np.abs(angles)) < self.params.angle_considered
                    # normal starts from the bottom side
                    valid_bottom_select = (np.abs(angles)) < self.params.angle_considered
                    if color in ['red', 'yellow']:
                        """
                        HW11-Step5
                        
                        HW11 Hints: Your codes will only use the following variables
                        mids_[top/bottom],normals_[top/bottom]: Both arrays have size Nx2, N represents the detected number of line segments
                        self.params.marker_width[color]: width of the color chunks
                        self.params.line_dist[color]: distance of the bottom edge to the parking center (BLACK CROSS ON THE MAP)
                        
                        Mids: Mid-points of the tangent vectors
                        Normals: Normal vectors heading to the color chunk
                        You are handling multiple detected lines at the same time, DON'T USE for-loop IF POSSIBLE (Try it)
                        
                        You need to calculate the distance to the parking center
                        If the car center is BELOW the parking center, the distance value should be NEGATIVE
                        If the car center is ABOVE the parking center, the distance value should be POSITIVE
                        dist2parking is an array with length N with all the distance you estimated using mids and normals
                        """

                        if valid_top_select.sum() > 0:
                            mids_top = mids[valid_top_select]
                            normals_top = normals[valid_top_select]
                            """
                            YOUR CODE HERE, Calculate the correct dist2parking with the same length as the normals_top
                            """
                            # Start >>> Your code here
                            dist2parking = np.zeros(len(normals_top))  # Fake result, write your own correct one
                            # End <<< Your code here
                            votes_dist.append(dist2parking)
                        if valid_bottom_select.sum() > 0:
                            mids_bottom = mids[valid_bottom_select]
                            normals_bottom = normals[valid_bottom_select]
                            """
                            YOUR CODE HERE, Calculate the correct dist2parking with the same length as the normals_bottom
                            """
                            # Start >>> Your code here
                            dist2parking = np.zeros(len(normals_bottom))  # Fake result, write your own correct one
                            # End <<< Your code here
                            votes_dist.append(dist2parking)
                    else:  # cyan, what is the difference, which side is closer to the center ?
                        if valid_top_select.sum() > 0:
                            mids_top = mids[valid_top_select]
                            normals_top = normals[valid_top_select]
                            """
                            YOUR CODE HERE, Calculate the correct dist2parking with the same length as the normals_top
                            """
                            # Start >>> Your code here
                            dist2parking = np.zeros(len(normals_top))  # Fake result, write your own correct one
                            # End <<< Your code here
                            votes_dist.append(dist2parking)
                        if valid_bottom_select.sum() > 0:
                            mids_bottom = mids[valid_bottom_select]
                            normals_bottom = normals[valid_bottom_select]
                            """
                            YOUR CODE HERE, Calculate the correct dist2parking with the same length as the normals_bottom
                            """
                            # Start >>> Your code here
                            dist2parking = np.zeros(len(normals_bottom))  # Fake result, write your own correct one
                            # End <<< Your code here
                            votes_dist.append(dist2parking)
                else:
                    warnings.warn(f'No valid {color} segments in robot frame')
                    continue
        if len(votes_dist) > 0:
            return np.concatenate(votes_dist)
        else:
            return []

    def get_normals(self, p1s, p2s, normal_types):
        t = p2s - p1s
        t = t / np.linalg.norm(t, axis=1).reshape(-1, 1)
        normals = np.hstack([t[:, [1]], -t[:, [0]]])  # Type1, flipping Z
        normals[normal_types == 2] *= -1  # Type2, flipping Z
        return normals

    def get_estimate(self):
        dist_idx = self.belief.argmax()
        dist_max = self.params.dist_min + (dist_idx + 0.5) * self.params.delta_dist
        if self.params.use_moving_avg:
            self.out_pred_buff = np.roll(self.out_pred_buff, 1)
            self.out_pred_buff[0] = dist_max
            return getattr(np, self.params.mov_avg_policy)(self.out_pred_buff)
        return dist_max

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
