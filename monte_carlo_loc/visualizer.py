#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: this script generate the evaluation results for mcl.

import loc_utils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib.patches import Ellipse


class Visualizer(object):
    """ This class is a visualizer for localization results.
    """
    def __init__(self, map_size, poses, gt_headings, numParticles=1000, start_idx=0, converge_thres=6):
        """ Initialization:
            map_size: the size of the given map
            poses: ground truth poses.
            numParticles: number of particles.
            start_idx: the start index.
            converge_thres: a threshold used to tell whether the localization converged or not.
        """
        self.numpoints = numParticles

        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots(3, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [5, 1, 1]})
        self.ax0 = self.ax[0]
        self.ax1 = self.ax[1]
        self.ax2 = self.ax[2]
        
        # set background image
        self.bg_img = None
        self.bg_img_extent = None  # xmin, xmax, ymin, ymax

        # set size
        self.point_size = 1
        self.map_size = map_size
        self.start_idx = start_idx
        self.converge_idx = start_idx

        # set ground truth
        N = len(poses)
        self.gt_locations = poses[:, :3, 3]  # 3D
        self.gt_headings = gt_headings  # 2D
        self.gt_rotations = np.identity(3).reshape(1, 3, 3).repeat(N, axis=0)  # 3D, for reg loc. Note: it's not identities in real app
        self.gt_translations = np.zeros((N, 3))  # 3D, for reg loc. Note: it's not zeros in real app

        # init estimates and errors
        self.location_estimates = np.zeros((N, 3))  # 3D
        self.heading_estimates = np.zeros(N)  # 2D
        self.location_err = np.zeros(N)  # 2D, in meter
        self.heading_err = np.zeros(N)  # 2D, in deg
        self.est_rotations = np.identity(3).reshape(1, 3, 3).repeat(N, axis=0)  # 3D, for reg loc
        self.est_translations = np.zeros((N, 3))  # 3D, for reg loc
        self.est_covariances = np.zeros((N, 4, 4))  # 3D, for reg loc
        self.est_reliable_values = np.zeros(N)
        self.AREs = np.zeros(N)  # in deg, for reg loc
        self.ATEs = np.zeros(N)  # in meter, for reg loc

        # Then setup FuncAnimation.
        self.err_thres = converge_thres
        self.setup_plot()

        # for zoom animation
        self.x_min_offset = 0
        self.x_max_offset = 0
        self.y_min_offset = 0
        self.y_max_offset = 0
        
        # status
        self.status = {
            'loc mode': 'PF',
            'is converged': False,
            'xy err': 0.0,
            'yaw err': 0.0,
            'sigma x': 0.0,
            'sigma y': 0.0,
            'sigma yaw': 0.0,
            'reliable value': 0.0  # λ
        }

    def setup_plot(self):
        """ Initial drawing of the scatter plot.
        """
        # setup ax0
        self.ax_gt, = self.ax0.plot([], [], c='r', label='reference pose')
        self.ax_est, = self.ax0.plot([], [], c='b', label='estimated pose')
        self.scat = self.ax0.scatter([], [], c=[], s=self.point_size, cmap="Blues", vmin=0, vmax=1)
        self.error_ellipse = Ellipse(xy=[0,0], width=1e-10, height=1e-10)
        
        if self.bg_img is not None and self.bg_img_extent is not None:
            self.ax0.imshow(self.bg_img, extent=self.bg_img_extent, alpha=0.9)

        self.ax0.axis('square')
        self.ax0.set(xlim=self.map_size[:2], ylim=self.map_size[2:])
        self.ax0.set_xlabel('X [m]')
        self.ax0.set_ylabel('Y [m]')
        self.ax0.legend()

        # setup ax1
        self.ax_location_err, = self.ax1.plot([], [], c='g')
        self.ax1.set(xlim=[0, len(self.gt_locations)], ylim=[0, self.err_thres])
        y_major_locator=MultipleLocator(2)
        self.ax1.yaxis.set_major_locator(y_major_locator)
        self.ax1.set_xlabel('Timestamp')
        self.ax1.set_ylabel('Location err [m]')
        self.ax1.grid()

        # setup ax2
        self.ax_heading_err, = self.ax2.plot([], [], c='g')
        self.ax2.set(xlim=[0, len(self.gt_locations)], ylim=[0, self.err_thres])
        y_major_locator=MultipleLocator(2)
        self.ax2.yaxis.set_major_locator(y_major_locator)
        self.ax2.set_xlabel('Timestamp')
        self.ax2.set_ylabel('Heading err [degree]')
        self.ax2.grid()

        # combine all artists
        self.patches = [self.ax_gt, self.ax_est, self.scat, self.error_ellipse, self.ax_location_err, self.ax_heading_err]

        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.patches

    def get_estimates(self, sorted_data, selection_rate=0.25):
        """ calculate the estimated poses.
        """
        # only use the top selection_rate particles to estimate the position
        selected_particles = sorted_data[-int(selection_rate * self.numpoints):]
        # normalize the weight
        normalized_weight = selected_particles[:, -1] / np.sum(selected_particles[:, -1])
        estimated_location = selected_particles[:, :-2].T.dot(normalized_weight.T)
        estimated_heading = selected_particles[:, -2].T.dot(normalized_weight.T)
        return estimated_location, estimated_heading

    def compute_errs_pf(self, frame_idx, particles):
        """ Calculate the errors.
        """
        sorted_data = particles[particles[:, -1].argsort()]
        new_location_estimate, new_heading_estimate = self.get_estimates(sorted_data)
        self.location_estimates[frame_idx] = new_location_estimate
        self.heading_estimates[frame_idx] = new_heading_estimate
        self.location_err[frame_idx] = np.linalg.norm(new_location_estimate[:2] - self.gt_locations[frame_idx, :2])
        self.heading_err[frame_idx] = abs(new_heading_estimate - self.gt_headings[frame_idx]) * 180. / np.pi
        return sorted_data
    
    def compute_errs_reg(self, frame_idx, query_pos, reg_state):
        # err
        if reg_state:  # use reg result
            est_R = self.est_rotations[frame_idx]
            est_t = self.est_translations[frame_idx]
            R_delta = np.matmul(est_R, np.linalg.inv(self.gt_rotations[frame_idx]))
            e = np.clip((np.trace(R_delta) - 1) / 2, -1, 1)
            self.AREs[frame_idx] = np.arccos(e) * 180.0 / np.pi  # [0, 180], in deg
            t_delta = est_t - self.gt_translations[frame_idx]
            self.ATEs[frame_idx] = np.linalg.norm(t_delta)
            query_pos = (est_R @ (query_pos.reshape(-1,1))).reshape(-1) + est_t
            self.location_estimates[frame_idx] = np.array([query_pos[0], query_pos[1], query_pos[2]])
            self.heading_estimates[frame_idx] = loc_utils.euler_angles_from_rotation_matrix(est_R)[2]
        else:  # use odometry result
            self.location_estimates[frame_idx] = np.array([query_pos[0], query_pos[1], query_pos[2]])
            self.heading_estimates[frame_idx] = query_pos[-1]
        self.location_err[frame_idx] = np.linalg.norm(self.location_estimates[frame_idx, :2] - self.gt_locations[frame_idx, :2])
        self.heading_err[frame_idx] = abs(self.heading_estimates[frame_idx] - self.gt_headings[frame_idx]) * 180. / np.pi
        # fake particle
        est_xyz = self.location_estimates[frame_idx]
        est_yaw = self.heading_estimates[frame_idx]
        fake_particles = np.array([[est_xyz[0], est_xyz[1], est_xyz[2], est_yaw, 1.0]])
        return fake_particles

    def update_status(self, status):
        for key in status:
            if key in self.status:
                self.status[key] = status[key]

    def update(self, frame_idx, particle_xyc):
        """ Update the scatter plot.
        """
        title = f"""Loc Mode: {self.status['loc mode']}, Converged: {self.status['is converged']}, Position Error: {self.status['xy err']:<.2f}m, Yaw Error: {self.status['yaw err']:<.2f}deg, λ: {self.status['reliable value']:<.5f}, Sigma(x,y,yaw): ({self.status['sigma x']:<.2f}m,{self.status['sigma y']:<.2f}m,{self.status['sigma yaw']:<.2f}deg)"""
        self.ax0.set_title(title, loc='center')
        # Only show the estimated trajectory when localization successfully converges
        if self.location_err[frame_idx] < self.err_thres:
            # set ground truth
            self.ax_gt.set_data(self.gt_locations[self.start_idx:frame_idx, 0],
                                self.gt_locations[self.start_idx:frame_idx, 1])

            # set estimated pose
            self.ax_est.set_data(self.location_estimates[self.converge_idx:frame_idx, 0],
                                 self.location_estimates[self.converge_idx:frame_idx, 1])

            # Set x and y data
            self.scat.set_offsets(particle_xyc[:, :2])
            # Set colors
            self.scat.set_array(particle_xyc[:, -1])

            # set err
            self.ax_location_err.set_data(np.arange(self.start_idx, frame_idx), self.location_err[self.start_idx:frame_idx])
            self.ax_heading_err.set_data(np.arange(self.start_idx, frame_idx), self.heading_err[self.start_idx:frame_idx])

        else:
            # set ground truth
            self.ax_gt.set_data(self.gt_locations[self.start_idx:frame_idx, 0],
                                self.gt_locations[self.start_idx:frame_idx, 1])

            # Set x and y data
            self.scat.set_offsets(particle_xyc[:, :2])
            # Set colors according to weights
            self.scat.set_array(particle_xyc[:, -1])

            # set err
            self.ax_location_err.set_data(np.arange(self.start_idx, frame_idx), self.location_err[self.start_idx:frame_idx])
            self.ax_heading_err.set_data(np.arange(self.start_idx, frame_idx), self.heading_err[self.start_idx:frame_idx])

            self.converge_idx += 1

        # We need to return the updated artist for FuncAnimation to draw.
        # Note that it expects a sequence of artists, thus the trailing comma.
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        return self.patches


if __name__ == '__main__':
    pass
