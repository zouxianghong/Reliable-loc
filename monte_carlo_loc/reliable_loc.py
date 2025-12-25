# Reliable localization module
# Created by Ericxhzou

import numpy as np
import os
import time
import matplotlib.pyplot as plt

import torch

from monte_carlo_loc.resample import resample
from monte_carlo_loc.sgv_utils import sgv_fn
from monte_carlo_loc.visualizer import Visualizer
from monte_carlo_loc.vis_loc_result import plot_traj_result, draw_traj_colored_by_loc_mode
from monte_carlo_loc import loc_utils
from utils.draw_result import draw_pps

from monte_carlo_loc.pf_loc import PFLoc
from monte_carlo_loc.reg_loc import RegLoc
from monte_carlo_loc.motion_model import MotionModel

class ReliableLoc:
    def __init__(self, logger, config, dataset, poses):
        # params
        self.log_dir = config['log_dir']
        self.start_idx = config['start_idx']
        self.end_idx = config['end_idx'] if config['end_idx'] < len(poses) else len(poses)
        self.num_particles = config['num_particles']
        self.dataset_name = config['dataset']
        self.loc_mode = config['loc_mode']
        self.max_sigma_x = config['max_sigma_x'] * 3
        self.max_sigma_y = config['max_sigma_y'] * 3
        self.max_sigma_yaw = config['max_sigma_yaw'] * 3
        self.check_reg_by_odometry = config['check_reg_by_odo']
        self.logger = logger
        # localizer
        self.pf_loc = PFLoc(logger, config, dataset, poses)
        self.reg_loc = RegLoc(logger, config, poses)
        # loc state
        self.use_regloc = False  # True: pf loc, False: reg loc
        self.loc_modes = []
        self.converge_states = []  # True: converged, False: not converged
        # latest robust frame idx
        self.robust_frame_idx = -1
        self.reg_states = np.zeros(len(poses), dtype=bool)
        # counts of not passing certication continuously
        self.not_pass_certify = 0
        self.max_not_pass_certify = 50  # max counts of not passing certication continuously
        # counts of passing certification continuously
        self.pass_certify = 0
        self.min_pass_certify = 3
        # counts of invalid registration in pf loc
        self.invalid_reg = 0
        self.max_invalid_reg = 50
        # time cost
        self.time_costs_reg = []
        # vis
        self.visualize = config['visualize']
        map_size = self.pf_loc.sensor_model.get_map_size()
        plt.ion()
        gt_headings = self.pf_loc.motion_model.gt_headings
        self.visualizer = Visualizer(map_size, poses, gt_headings,
                                        numParticles=config['num_particles'],
                                        start_idx=self.start_idx)
        self.pf_loc.visualizer = self.visualizer
        # save result
        self.loc_results = self.pf_loc.loc_results
    
    def get_loc_mode(self):
        if self.use_regloc:
            return 'reg_loc'
        else:
            return 'pf_loc'
    
    def get_xyzyaw_by_odometry(self, start_frame_idx, frame_idx):
        start_xyz = self.visualizer.location_estimates[start_frame_idx]
        start_yaw = self.visualizer.heading_estimates[start_frame_idx]
        xyzyaw = np.array([[start_xyz[0], start_xyz[1], start_xyz[2], start_yaw]])
        for i in range(start_frame_idx, frame_idx):
            xyzyaw = self.pf_loc.motion_model.move(xyzyaw, i)
        return xyzyaw.reshape(-1)

    def get_xyzyaw_cov_by_odometry(self, frame_idx):
        ''' @Input: frame_idx
            @Return: xyzyaw: [est x, est y, est_z, est yaw], cov1: 3 x 3
        '''
        start_frame_idx = self.robust_frame_idx
        if start_frame_idx == -1:
            return None, None
        # start
        start_xyz = self.visualizer.location_estimates[start_frame_idx]
        start_yaw = self.visualizer.heading_estimates[start_frame_idx]
        xyzyaw = np.array([start_xyz[0], start_xyz[1], start_xyz[2], start_yaw])
        cov = self.visualizer.est_covariances[start_frame_idx]
        cov1 = np.zeros((3, 3))
        cov1[:2, :2], cov1[:2, 2] = cov[:2, :2], cov[:2, 3]
        cov1[2, :2], cov1[2, 2] = cov[3, :2], cov[3, 3]
        cov2 = np.zeros((3, 3))
        cov2[0,0] = self.pf_loc.motion_model.noise_dist**2
        cov2[1,1] = self.pf_loc.motion_model.noise_rot1**2
        cov2[2,2] = self.pf_loc.motion_model.noise_rot2**2
        for i in range(start_frame_idx, frame_idx):
            dist = self.pf_loc.motion_model.commands[i, 1]
            rot1 = self.pf_loc.motion_model.commands[i, 0]
            rot2 = self.pf_loc.motion_model.commands[i, 2]
            cov1 = MotionModel.propagate_cov(xyzyaw[-1], dist, rot1, rot2, cov1, cov2)
            xyzyaw = self.get_xyzyaw_by_odometry(start_frame_idx, i+1)
        return xyzyaw, cov1
    
    def switch_to_pf(self, frame_idx, use_cov=False, num_particles=400):
        xyzyaw, cov = self.get_xyzyaw_cov_by_odometry(frame_idx)
        if use_cov and cov is not None:
            self.pf_loc.particles = self.pf_loc.sensor_model.init_particles(
                num_particles=num_particles, init_type='cov', x=xyzyaw[0], y=xyzyaw[1], z=xyzyaw[2], yaw=xyzyaw[3], cov=cov)
        else:
            xyzyaw = self.get_xyzyaw_by_odometry(self.robust_frame_idx, frame_idx)
            self.pf_loc.particles = self.pf_loc.sensor_model.init_particles(
                num_particles=num_particles, init_type='radius_yaw', x=xyzyaw[0], y=xyzyaw[1], z=xyzyaw[2], yaw=xyzyaw[3], radius=10.0)
    
    def pairwise_reg(self, frame_idx, vis_pps=False):
        t0_reg = time.time()
        is_valid, q_kpts, m_kpts, est_R, est_t, est_theta = False, None, None, None, None, None
        q_l_kpt, q_l_feat = self.pf_loc.dataset.get_l_kpt_desc(
            self.pf_loc.sensor_model.model_type,
            self.pf_loc.sensor_model.query_indices[frame_idx],
            unify_coord=True, g_offset=False)
        if q_l_kpt is not None and q_l_feat is not None:
            # closest map grid
            start_frame_idx = -1
            if self.use_regloc:
                start_frame_idx = self.robust_frame_idx
            if start_frame_idx == -1 or frame_idx - start_frame_idx > 100:
                start_frame_idx = frame_idx-1
            xyzyaw = self.get_xyzyaw_by_odometry(start_frame_idx, frame_idx)
            est_xy = np.array([xyzyaw[0], xyzyaw[1]])
            dists, indices = self.pf_loc.sensor_model.map_pos_tree.query(est_xy.reshape(1, -1), k=5)
            map_grid_idx, max_dist = -1, 30.0
            for j in range(len(indices[0])):
                if dists[0][j] > max_dist:
                    break
                map_grid_idx = self.pf_loc.sensor_model.map_indices[indices[0][j]]
            if map_grid_idx != -1:
                # map grid
                m_l_kpt, m_l_feat = self.pf_loc.dataset.get_l_kpt_desc(self.pf_loc.sensor_model.model_type, map_grid_idx, unify_coord=True, g_offset=False)
                if m_l_kpt is not None and m_l_feat is not None:
                    # tensor
                    init_q_kpts, init_m_kpts = torch.from_numpy(q_l_kpt[None, ...]), torch.from_numpy(m_l_kpt[None, ...])  # 1 x k x 3
                    init_q_feats, init_m_feats = torch.from_numpy(q_l_feat[None, ...]), torch.from_numpy(m_l_feat[None, ...])  # 1 x k x d
                    # correspondence Estimation: Nearest Neighbour Matching
                    mask = None
                    # FIXME: sth wrong with the code ?
                    # if self.robust_frame_idx != -1 and frame_idx - self.robust_frame_idx < 20:
                    #     xyyaw, cov = self.get_xyyaw_cov_by_odometry(frame_idx)
                    #     init_R = loc_utils.rotation_matrix_from_euler_angles(yaw=xyyaw[2], degrees=False)
                    #     init_t = xyyaw[:2] - self.pf_loc.poses[frame_idx, :2, 3]
                    #     init_t = np.array([init_t[0], init_t[1],0.0])
                    #     transformed_q_l_kpt = q_l_kpt@init_R.T + init_t.T
                    #     draw_pps(transformed_q_l_kpt, m_l_kpt, offset_x=None)
                    #     delta_xy = np.abs(transformed_q_l_kpt[:, None, :2] - m_l_kpt[None, :, :2])  # k x k x 2
                    #     sin_yaw, cos_yaw = np.sin(xyyaw[2]), np.cos(xyyaw[2])
                    #     sigma_xy = []
                    #     for i in range(q_l_kpt.shape[0]):
                    #         J_i = np.array([
                    #             [1, 0, -sin_yaw*q_l_kpt[i,0]-cos_yaw*q_l_kpt[i,1]],
                    #             [0, 1, cos_yaw*q_l_kpt[i,0]-sin_yaw*q_l_kpt[i,1]]
                    #         ])
                    #         cov_xy_i = np.matmul(J_i, np.matmul(cov, J_i.T))
                    #         sigma_xy.append(np.sqrt(np.diag(cov_xy_i)) * 3)
                    #     sigma_xy = np.array(sigma_xy)  # k x 2
                    #     mask = np.where(delta_xy - sigma_xy[:, None, :] > 0.0, 1, 0)  # k x k
                    #     mask = None
                    conf_list, q_kpts, m_kpts, weights = sgv_fn(init_q_feats, init_q_kpts, init_m_feats, init_m_kpts, mask=mask, k=256)
                    q_kpts, m_kpts = q_kpts[0], m_kpts[0]
                    # registration
                    est_R, est_t, est_theta = self.reg_loc.reg_with_teaser(q_kpts, m_kpts)
                    # validation state
                    is_valid = np.sum(est_theta == 1) >= 3
                    # covariance and degeneration value
                    if is_valid:
                        cov, reliable_value = RegLoc.get_cov(q_kpts, est_R, est_theta)
                        self.visualizer.est_covariances[frame_idx] = cov  # 4 x 4
                        self.visualizer.est_reliable_values[frame_idx] = reliable_value
                    # for debug: vis pps
                    if vis_pps and is_valid:
                        from utils.util import transform_points
                        # draw_pps(q_kpts, m_kpts)
                        valid_q_kpts, valid_m_kpts = q_kpts[est_theta == 1], m_kpts[est_theta == 1]
                        title = f'frame: {frame_idx}, pps: {np.sum(est_theta == 1)}, reliable: {self.visualizer.est_reliable_values[frame_idx]:<.6f}'
                        draw_pps(valid_q_kpts, valid_m_kpts, title=title)
                        # transformed_q_kpts = transform_points(valid_q_kpts, est_R) + est_t.reshape(1, -1)
                        # draw_pps(transformed_q_kpts, valid_m_kpts, offset_x=None)
                        print()
        t1_reg = time.time()
        time_cost_reg = (t1_reg - t0_reg) * 1000
        self.time_costs_reg.append(time_cost_reg)
        # self.logger.info(f'reg time cost: {time_cost_reg:<.2f}ms, reg state: {is_valid}')
        return is_valid, q_kpts, m_kpts, est_R, est_t, est_theta
    
    def check_reg_by_odo(self, frame_idx, est_R, est_t):
        pass_check = True
        if self.robust_frame_idx != -1 and frame_idx - self.robust_frame_idx < 50:
            # 2d pose estimated by odometry
            xyzyaw, cov = self.get_xyzyaw_cov_by_odometry(frame_idx)
            sigma_xyyaw = np.sqrt(np.diag(cov))
            # 2d pose estimated by reg
            query_pos = self.pf_loc.poses[frame_idx, :3, 3]
            query_pos = (est_R @ (query_pos.reshape(-1,1))).reshape(-1) + est_t
            yaw2 = loc_utils.euler_angles_from_rotation_matrix(est_R)[2]
            xyzyaw2 = np.array([query_pos[0], query_pos[1], query_pos[2], yaw2])
            # check
            delta_xyzyaw = np.abs(xyzyaw - xyzyaw2)
            pass_check = delta_xyzyaw[0] < 3 * sigma_xyyaw[0] and delta_xyzyaw[1] < 3 * sigma_xyyaw[1] and delta_xyzyaw[-1] < 3 * sigma_xyyaw[2]
        return pass_check
    
    def run_pf_loc(self):
        ''' loc mode: 0, use particle filter only'''
        time_costs = []
        for frame_idx in range(self.start_idx, self.end_idx):
            t0 = time.time()
            # estimate pose and visualize
            particle_xyc = self.visualizer.compute_errs_pf(frame_idx, self.pf_loc.particles)
            if self.visualize:
                self.visualizer.update(frame_idx, particle_xyc)
            self.loc_results[frame_idx, :len(self.pf_loc.particles)] = self.pf_loc.particles
            # move particles and update their weights
            self.pf_loc.run_one_step(frame_idx)
            # initialize again if no valid particles (all particles go beyond the base map)
            if self.pf_loc.particles is None:
                self.pf_loc.particles = self.pf_loc.sensor_model.init_particles(self.pf_loc.num_particles)
                self.logger.info(f'Initialize particles! frame: {frame_idx}')
            self.converge_states.append(self.pf_loc.sensor_model.is_converged)
            # time cost
            t1 = time.time()
            time_cost = (t1 - t0) * 1000  # unit: ms
            time_costs.append(time_cost)
            reg_state = self.reg_states[frame_idx]
            dist_err = self.visualizer.location_err[frame_idx]
            yaw_err = self.visualizer.heading_err[frame_idx]
            self.logger.info(f'Frame: {frame_idx}, dist err: {dist_err:<.2f}, yaw err: {yaw_err:<.2f}, reg state: {reg_state}, time cost: {time_cost:<.2f}ms')
        # converge ratio
        converge_ratio = np.sum(self.converge_states) / len(self.converge_states) * 100
        self.logger.info(f'[Converge Ratio]: {converge_ratio:<.2f}%')
        # time cost
        self.pf_loc.log_time_cost()
        mean_time_cost = np.mean(time_costs)
        std_time_cost = np.std(time_costs)
        self.logger.info(f'Total time cost: {mean_time_cost:<.2f} ± {std_time_cost:<.2f}ms')
        # save loc result
        res = {
            'particles': self.loc_results,
            'loc_modes': self.loc_modes,
            'converge_states': self.converge_states,
            'num_occupancies': self.pf_loc.num_occupancies[self.start_idx:self.end_idx]
        }
        result_file = os.path.join(self.log_dir, f'localization_results_{self.dataset_name}_{self.start_idx}_{self.end_idx}')
        np.savez_compressed(result_file, res)
        # fig vis
        if self.visualize:
            fig_vis = os.path.join(self.log_dir, f'{self.dataset_name}_vis.jpg')
            self.visualizer.fig.savefig(fig_vis, transparent=False, bbox_inches='tight')
        # fig traj
        fig_traj = os.path.join(self.log_dir, f'{self.dataset_name}_traj.jpg')
        plot_traj_result(res, self.pf_loc.poses, self.pf_loc.motion_model.gt_headings, numParticles=self.num_particles,
                        start_idx=self.start_idx, end_idx=self.end_idx,
                        save_filepath=fig_traj, logger=self.logger)
        # fig converge states
        fig_conv_state = os.path.join(self.log_dir, f'{self.dataset_name}_conv_state.png')
        draw_traj_colored_by_loc_mode(self.converge_states, self.pf_loc.poses[self.start_idx:self.end_idx], save_filepath=fig_conv_state)
    
    def run_reg_loc(self):
        ''' loc mode: -1, use reg only'''
        time_costs = []
        for frame_idx in range(self.start_idx, self.end_idx):
            self.loc_modes.append(self.get_loc_mode())
            t0 = time.time()
            # run pf loc or reg loc
            if self.pf_loc.sensor_model.is_converged:  # judge pf's convergency or loc state
                # register and try to send certification request
                is_valid, q_kpts, m_kpts, est_R, est_t, est_theta = self.pairwise_reg(frame_idx, vis_pps=False)
                # use odometry to check reg result
                if is_valid and self.check_reg_by_odometry:
                    is_valid = self.check_reg_by_odo(frame_idx, est_R, est_t)
                # if frame_idx < self.start_idx + 5:  # FIXME: for extra experiment
                #     is_valid = True
                if is_valid:  # registration is valid so that a certification request is sent
                    self.reg_states[frame_idx] = True
                    self.visualizer.est_rotations[frame_idx] = est_R
                    self.visualizer.est_translations[frame_idx] = est_t
                    if self.reg_loc.use_certify_algo:
                        request = {
                            'frame_idx': frame_idx,
                            'type': 'certify',
                            'q_kpts': q_kpts,
                            'm_kpts': m_kpts,
                            'est_R': est_R,
                            'est_theta': est_theta
                        }
                    else:
                        request = {
                            'frame_idx': frame_idx,
                            'type': 'certify',
                            'cov': self.visualizer.est_covariances[frame_idx],
                            'reliable_value': self.visualizer.est_reliable_values[frame_idx]
                        }
                    # certify
                    result = self.reg_loc.run_certify_once(request)
                    # if frame_idx < self.start_idx + 5:  # FIXME: for extra experiment
                    #     result['pass_certify'] = True
                    # deal with certification results
                    if result['pass_certify']:  # pass certification
                        self.pass_certify += 1
                        if self.pass_certify >=self.min_pass_certify:
                            if not self.use_regloc:
                                self.logger.info('Switch to Reg loc!')
                                self.use_regloc = True
                            self.pf_loc.sensor_model.count_invalid = 0
                            self.robust_frame_idx = result['frame_idx']
                    else:  # fail to pass certification
                        self.pass_certify = 0
                else:
                    self.pass_certify = 0
            # use reg loc
            if self.use_regloc:
                # estimate pose and visualize
                use_reg_res = self.robust_frame_idx == frame_idx
                if use_reg_res:  # use pose estimated by reg
                    query_pos = self.pf_loc.poses[frame_idx, :3, 3]  # Note: should use real query submap center in practical applications
                else:  # use pose provided by odometry
                    start_frame_idx = self.robust_frame_idx
                    if start_frame_idx == -1 or frame_idx - self.robust_frame_idx > 100:
                        start_frame_idx = frame_idx-1
                    query_pos = self.get_xyzyaw_by_odometry(start_frame_idx, frame_idx)
                particle_xyc = self.visualizer.compute_errs_reg(
                    frame_idx, query_pos, reg_state=use_reg_res)  # fake particle for vis
                if self.visualize:
                    self.visualizer.update(frame_idx, particle_xyc)
                self.loc_results[frame_idx, :len(particle_xyc)] = particle_xyc
            else:  # use pf loc
                # estimate pose and visualize
                particle_xyc = self.visualizer.compute_errs_pf(frame_idx, self.pf_loc.particles)
                if self.visualize:
                    self.visualizer.update(frame_idx, particle_xyc)
                self.loc_results[frame_idx, :len(self.pf_loc.particles)] = self.pf_loc.particles
                # move particles and update their weights
                self.pf_loc.run_one_step(frame_idx)
                # initialize again if no valid particles (all particles go beyond the base map)
                if self.pf_loc.particles is None:
                    self.pf_loc.particles = self.pf_loc.sensor_model.init_particles(self.pf_loc.num_particles)
                    self.logger.info(f'Initialize particles! frame: {frame_idx}')
            self.converge_states.append(self.pf_loc.sensor_model.is_converged)
            # time cost
            t1 = time.time()
            time_cost = (t1 - t0) * 1000  # unit: ms
            time_costs.append(time_cost)
            reg_state = self.reg_states[frame_idx]
            dist_err = self.visualizer.location_err[frame_idx]
            yaw_err = self.visualizer.heading_err[frame_idx]
            self.logger.info(f'Frame: {frame_idx}, dist err: {dist_err:<.2f}, yaw err: {yaw_err:<.2f}, reg state: {reg_state}, time cost: {time_cost:<.2f}ms')
        # converge ratio
        converge_ratio = np.sum(self.converge_states) / len(self.converge_states) * 100
        self.logger.info(f'[Converge Ratio]: {converge_ratio:<.2f}%')
        # time cost
        self.pf_loc.log_time_cost()
        mean_time_cost = np.mean(self.time_costs_reg)
        std_time_cost = np.std(self.time_costs_reg)
        self.logger.info(f'[Reg time cost]: {mean_time_cost:<.2f} ± {std_time_cost:<.2f}ms')
        mean_time_cost = np.mean(time_costs)
        std_time_cost = np.std(time_costs)
        self.logger.info(f'[Total time cost]: {mean_time_cost:<.2f} ± {std_time_cost:<.2f}ms')
        # save loc result
        res = {
            'particles': self.loc_results,
            'loc_modes': self.loc_modes,
            'converge_states': self.converge_states,
            'num_occupancies': self.pf_loc.num_occupancies[self.start_idx:self.end_idx]
        }
        result_file = os.path.join(self.log_dir, f'localization_results_{self.dataset_name}_{self.start_idx}_{self.end_idx}')
        np.savez_compressed(result_file, res)
        # fig vis
        if self.visualize:
            fig_vis = os.path.join(self.log_dir, f'{self.dataset_name}_vis.jpg')
            self.visualizer.fig.savefig(fig_vis, transparent=False, bbox_inches='tight')
        # fig traj
        fig_traj = os.path.join(self.log_dir, f'{self.dataset_name}_traj.jpg')
        plot_traj_result(res, self.pf_loc.poses, self.pf_loc.motion_model.gt_headings, numParticles=self.num_particles,
                        start_idx=self.start_idx, end_idx=self.end_idx,
                        save_filepath=fig_traj, logger=self.logger)
        # fig loc modes
        fig_loc_mode = os.path.join(self.log_dir, f'{self.dataset_name}_loc_mode.png')
        draw_traj_colored_by_loc_mode(np.array(self.loc_modes) == 'reg_loc', self.pf_loc.poses[self.start_idx:self.end_idx], save_filepath=fig_loc_mode)
        # fig converge states
        fig_conv_state = os.path.join(self.log_dir, f'{self.dataset_name}_conv_state.png')
        draw_traj_colored_by_loc_mode(self.converge_states, self.pf_loc.poses[self.start_idx:self.end_idx], save_filepath=fig_conv_state)
    
    def run_reliable_loc(self):
        time_costs = []
        for frame_idx in range(self.start_idx, self.end_idx):
            self.loc_modes.append(self.get_loc_mode())
            t0 = time.time()
            # run pf loc or reg loc
            if self.pf_loc.sensor_model.is_converged:  # judge pf's convergency or loc state
                # register and try to send certification request
                is_valid, q_kpts, m_kpts, est_R, est_t, est_theta = self.pairwise_reg(frame_idx, vis_pps=False)
                # use odometry to check reg result
                if is_valid and self.check_reg_by_odometry:
                    is_valid = self.check_reg_by_odo(frame_idx, est_R, est_t)
                # if frame_idx < self.start_idx + 5:  # FIXME: for extra experiment
                #     is_valid = True
                if is_valid:  # registration is valid so that a certification request is sent
                    self.reg_states[frame_idx] = True
                    self.visualizer.est_rotations[frame_idx] = est_R
                    self.visualizer.est_translations[frame_idx] = est_t
                    if self.reg_loc.use_certify_algo:
                        request = {
                            'frame_idx': frame_idx,
                            'type': 'certify',
                            'q_kpts': q_kpts,
                            'm_kpts': m_kpts,
                            'est_R': est_R,
                            'est_theta': est_theta
                        }
                    else:
                        request = {
                            'frame_idx': frame_idx,
                            'type': 'certify',
                            'cov': self.visualizer.est_covariances[frame_idx],
                            'reliable_value': self.visualizer.est_reliable_values[frame_idx]
                        }
                        if request['reliable_value'] < 1e-4:
                            self.invalid_reg += 1
                        else:
                            self.invalid_reg = 0
                    # certify
                    result = self.reg_loc.run_certify_once(request)
                    # if frame_idx < self.start_idx + 5:  # FIXME: for extra experiment
                    #     result['pass_certify'] = True
                    # deal with certification results
                    if result['pass_certify']:  # pass certification
                        self.pass_certify += 1
                        if self.pass_certify >=self.min_pass_certify:
                            if not self.use_regloc:
                                self.logger.info('Switch to Reg loc!')
                                self.use_regloc = True
                            self.not_pass_certify = 0
                            self.pf_loc.sensor_model.count_invalid = 0
                            self.robust_frame_idx = result['frame_idx']
                    else:  # fail to pass certification
                        self.pass_certify = 0
                        self.not_pass_certify += 1
                        if self.use_regloc:  # use pf loc and re-initialize particles
                            # switch to pf loc if covariance is too large
                            _, cov = self.get_xyzyaw_cov_by_odometry(frame_idx)
                            sigma_xyyaw = np.sqrt(np.diag(cov))
                            switch = sigma_xyyaw[0] > self.max_sigma_x or sigma_xyyaw[1] > self.max_sigma_y or sigma_xyyaw[2]*180/np.pi > self.max_sigma_yaw
                            if switch or self.not_pass_certify == self.max_not_pass_certify:
                                self.logger.info('Switch to PF loc and re-initialize particles!')
                                self.switch_to_pf(frame_idx, use_cov=True)
                                self.use_regloc = False
                                self.not_pass_certify = 0
                else:
                    self.pass_certify = 0
                    self.not_pass_certify += 1
                    if self.use_regloc:  # switch to pf loc and re-initialize particles when registration is failed
                        self.invalid_reg = 0
                        # switch to pf loc if covariance is too large
                        _, cov = self.get_xyzyaw_cov_by_odometry(frame_idx)
                        sigma_xyyaw = np.sqrt(np.diag(cov))
                        switch = sigma_xyyaw[0] > self.max_sigma_x or sigma_xyyaw[1] > self.max_sigma_y or sigma_xyyaw[2]*180/np.pi > self.max_sigma_yaw
                        if switch or self.not_pass_certify == self.max_not_pass_certify:
                            self.logger.info('Switch to PF loc and re-initialize particles!')
                            self.switch_to_pf(frame_idx, use_cov=True)
                            self.use_regloc = False
                            self.not_pass_certify = 0
                    else:
                        self.invalid_reg += 1
            # use reg loc
            if self.use_regloc:
                # update error ellipse
                _, cov = self.get_xyzyaw_cov_by_odometry(frame_idx)
                # estimate pose and visualize
                use_reg_res = self.robust_frame_idx == frame_idx
                if use_reg_res:  # use pose estimated by reg
                    query_pos = self.pf_loc.poses[frame_idx, :3, 3]  # Note: should use real query submap center in practical applications
                else:  # use pose estimated by odometry
                    start_frame_idx = self.robust_frame_idx
                    if start_frame_idx == -1 or frame_idx - self.robust_frame_idx > 100:
                        start_frame_idx = frame_idx-1
                    query_pos = self.get_xyzyaw_by_odometry(start_frame_idx, frame_idx)
                particle_xyc = self.visualizer.compute_errs_reg(
                    frame_idx, query_pos, reg_state=use_reg_res)  # fake particle for vis
                if self.visualize:
                    self.visualizer.update_status(status={
                        'loc mode': 'PF' if self.get_loc_mode() == 'pf_loc' else 'Reg',
                        'is converged': self.pf_loc.sensor_model.is_converged,
                        'xy err': self.visualizer.location_err[frame_idx],
                        'yaw err': self.visualizer.heading_err[frame_idx],
                        'sigma x': 0.0 if cov is None else np.sqrt(cov[0,0]),
                        'sigma y': 0.0 if cov is None else np.sqrt(cov[1,1]),
                        'sigma yaw': 0.0 if cov is None else np.sqrt(cov[2,2])*180/np.pi,
                        'reliable value': self.visualizer.est_reliable_values[frame_idx]
                    })
                    self.visualizer.update(frame_idx, particle_xyc)
                self.loc_results[frame_idx, :len(particle_xyc)] = particle_xyc
            else:  # use pf loc
                # estimate pose and visualize
                particle_xyc = self.visualizer.compute_errs_pf(frame_idx, self.pf_loc.particles)
                if self.visualize:
                    cov = self.visualizer.est_covariances[frame_idx]
                    self.visualizer.update_status(status={
                        'loc mode': 'PF' if self.get_loc_mode() == 'pf_loc' else 'Reg',
                        'is converged': self.pf_loc.sensor_model.is_converged,
                        'xy err': self.visualizer.location_err[frame_idx],
                        'yaw err': self.visualizer.heading_err[frame_idx],
                        'sigma x': 0.0 if cov is None else np.sqrt(cov[0,0]),
                        'sigma y': 0.0 if cov is None else np.sqrt(cov[1,1]),
                        'sigma yaw': 0.0 if cov is None else np.sqrt(cov[3,3])*180/np.pi,
                        'reliable value': self.visualizer.est_reliable_values[frame_idx]
                    })
                    self.visualizer.update(frame_idx, particle_xyc)
                self.loc_results[frame_idx, :len(self.pf_loc.particles)] = self.pf_loc.particles
                # move particles and update their weights
                self.pf_loc.run_one_step(frame_idx)
                # initialize again if no valid particles (all particles go beyond the base map)
                if self.pf_loc.particles is None:
                    self.logger.info(f'[{self.get_loc_mode()}] Re-initialize particles! frame: {frame_idx}')
                    if self.robust_frame_idx != -1 and frame_idx - self.robust_frame_idx < 200:
                        _, cov = self.get_xyzyaw_cov_by_odometry(frame_idx)
                    else:
                        cov = None
                    if cov is None:
                        self.pf_loc.particles = self.pf_loc.sensor_model.init_particles(self.pf_loc.num_particles)
                    else:
                        self.switch_to_pf(frame_idx, use_cov=True)
                    self.not_pass_certify = 0
                    self.invalid_reg = 0
                # elif self.invalid_reg >= self.max_invalid_reg:  # FIXME: is it necessary?
                #     if self.robust_frame_idx != -1 and frame_idx - self.robust_frame_idx < 200:
                #         _, cov = self.get_xyzyaw_cov_by_odometry(frame_idx)
                #         if cov is not None:
                #             self.logger.info(f'[{self.get_loc_mode()}] Re-initialize particles! frame: {frame_idx}')
                #             self.switch_to_pf(frame_idx, use_cov=True)
                #             self.not_pass_certify = 0
                #             self.invalid_reg = 0
            self.converge_states.append(self.pf_loc.sensor_model.is_converged)
            # time cost
            t1 = time.time()
            time_cost = (t1 - t0) * 1000  # unit: ms
            time_costs.append(time_cost)
            reg_state = self.reg_states[frame_idx]
            dist_err = self.visualizer.location_err[frame_idx]
            yaw_err = self.visualizer.heading_err[frame_idx]
            self.logger.info(f'[{self.get_loc_mode()}] Frame: {frame_idx}, dist err: {dist_err:<.2f}, yaw err: {yaw_err:<.2f}, reg state: {reg_state}, time cost: {time_cost:<.2f}ms')
        # converge ratio
        converge_ratio = np.sum(self.converge_states) / len(self.converge_states) * 100
        self.logger.info(f'[Converge Ratio]: {converge_ratio:<.2f}%')
        # time cost
        self.pf_loc.log_time_cost()
        mean_time_cost = np.mean(self.time_costs_reg)
        std_time_cost = np.std(self.time_costs_reg)
        self.logger.info(f'[Reg time cost]: {mean_time_cost:<.2f} ± {std_time_cost:<.2f}ms')
        mean_time_cost = np.mean(time_costs)
        std_time_cost = np.std(time_costs)
        self.logger.info(f'[Total time cost]: {mean_time_cost:<.2f} ± {std_time_cost:<.2f}ms')
        # loc mode analysis
        num_reg_loc = 0
        for mode in self.loc_modes:
            if mode == 'reg_loc':
                num_reg_loc += 1
        self.logger.info(f'Loc Mode Analysis, reg_loc / total: {num_reg_loc} / {len(self.loc_modes)}')
        # save loc result
        res = {
            'particles': self.loc_results,
            'loc_modes': self.loc_modes,
            'converge_states': self.converge_states,
            'num_occupancies': self.pf_loc.num_occupancies[self.start_idx:self.end_idx]
        }
        result_file = os.path.join(self.log_dir, f'localization_results_{self.dataset_name}_{self.start_idx}_{self.end_idx}')
        np.savez_compressed(result_file, res)
        # fig vis
        if self.visualize:
            fig_vis = os.path.join(self.log_dir, f'{self.dataset_name}_vis.jpg')
            self.visualizer.fig.savefig(fig_vis, transparent=False, bbox_inches='tight')
        # fig traj
        fig_traj = os.path.join(self.log_dir, f'{self.dataset_name}_traj.jpg')
        plot_traj_result(res, self.pf_loc.poses, self.pf_loc.motion_model.gt_headings, numParticles=self.num_particles,
                        start_idx=self.start_idx, end_idx=self.end_idx,
                        save_filepath=fig_traj, logger=self.logger)
        # fig loc modes
        fig_loc_mode = os.path.join(self.log_dir, f'{self.dataset_name}_loc_mode.png')
        draw_traj_colored_by_loc_mode(np.array(self.loc_modes) == 'reg_loc', self.pf_loc.poses[self.start_idx:self.end_idx], save_filepath=fig_loc_mode)
        # fig converge states
        fig_conv_state = os.path.join(self.log_dir, f'{self.dataset_name}_conv_state.png')
        draw_traj_colored_by_loc_mode(self.converge_states, self.pf_loc.poses[self.start_idx:self.end_idx], save_filepath=fig_conv_state)
    
    def run_test(self, vis_pps=True):
        ''' test teaser++ / cov and analyse params
        '''
        AREs, ATEs, sigma_xs, sigma_ys, sigma_yaws, reliable_values = [], [], [], [], [], []
        for frame_idx in range(self.start_idx, self.end_idx):
            q_l_kpt, q_l_feat = self.pf_loc.dataset.get_l_kpt_desc(
                self.pf_loc.sensor_model.model_type,
                self.pf_loc.sensor_model.query_indices[frame_idx],
                unify_coord=True, g_offset=False)
            if q_l_kpt is None or q_l_feat is None:
                 continue
            q_xy = self.pf_loc.poses[frame_idx, :2, 3]
            indices = self.pf_loc.sensor_model.map_pos_tree.query_radius(q_xy.reshape(1, -1), r=30)
            for j in indices[0]:
                map_grid_idx = self.pf_loc.sensor_model.map_indices[j]
                m_l_kpt, m_l_feat = self.pf_loc.dataset.get_l_kpt_desc(
                    self.pf_loc.sensor_model.model_type, map_grid_idx, unify_coord=True, g_offset=False)
                if m_l_kpt is None or m_l_feat is None:
                    continue
                # tensor
                q_kpts, m_kpts = torch.from_numpy(q_l_kpt[None, ...]), torch.from_numpy(m_l_kpt[None, ...])  # 1 x k x 3
                q_feats, m_feats = torch.from_numpy(q_l_feat[None, ...]), torch.from_numpy(m_l_feat[None, ...])  # 1 x k x d
                # correspondence Estimation: Nearest Neighbour Matching
                conf_list, q_kpts, m_kpts, weights = sgv_fn(q_feats, q_kpts, m_feats, m_kpts, k=256)
                q_kpts, m_kpts = q_kpts[0], m_kpts[0]
                # registration
                est_R, est_t, est_theta = self.reg_loc.reg_with_teaser(q_kpts, m_kpts)
                # ATE / ARE
                if np.sum(est_theta==1) >=3:
                    R_delta = np.matmul(est_R, np.linalg.inv(np.identity(3)))
                    e = np.clip((np.trace(R_delta) - 1) / 2, -1, 1)
                    ARE = np.arccos(e) * 180.0 / np.pi  # [0, 180], in deg
                    t_delta = est_t - np.zeros(3)
                    ATE = np.linalg.norm(t_delta)
                    AREs.append(ARE)
                    ATEs.append(ATE)
                    self.logger.info(f'ARE: {ARE}, ATE: {ATE}')
                # covariance
                if np.sum(est_theta==1) >= 3:
                    cov, reliable_value = self.reg_loc.get_cov(q_kpts, est_R, est_theta)
                    sigma_xyzyaw = np.sqrt(np.diag(cov))
                    sigma_xs.append(sigma_xyzyaw[0])
                    sigma_ys.append(sigma_xyzyaw[1])
                    sigma_yaws.append(sigma_xyzyaw[3]*180/np.pi)
                    reliable_values.append(reliable_value)
                    self.logger.info(f'Certify frame: {frame_idx}, sigma: [{sigma_xyzyaw[0]}, {sigma_xyzyaw[1]}, {sigma_xyzyaw[3]*180/np.pi}], reliable_value: {reliable_value}')
                # certify
                if np.sum(est_theta==1) >= 3 and False:
                    certify_q_kpts = q_kpts[est_theta == 1]
                    certify_m_kpts = m_kpts[est_theta == 1]
                    certify_theta = np.ones(len(certify_q_kpts))
                    is_optimal, best_suboptimality = self.reg_loc.certify(certify_q_kpts, certify_m_kpts, est_R, certify_theta)
                    self.logger.info(f'Certify frame: {frame_idx}, is_optimal: {is_optimal}, suboptimality: {best_suboptimality}')
                # for debug: vis pps
                if vis_pps:
                    from utils.util import transform_points
                    # draw_pps(q_kpts, m_kpts, offset_x=None)
                    if np.sum(est_theta == 1) == 0:
                        continue
                    valid_q_kpts, valid_m_kpts = q_kpts[est_theta == 1], m_kpts[est_theta == 1]
                    draw_pps(valid_q_kpts, valid_m_kpts)
                    # transformed_q_kpts = transform_points(valid_q_kpts, est_R) + est_t.reshape(1, -1)
                    # draw_pps(transformed_q_kpts, valid_m_kpts, offset_x=None)
                    print()
        # statistics, 5deg, 2m
        import math
        ARE_thresh, ATE_thresh, reliable_thresh = 5, 2, 1e-4
        ARE_bins, ATE_bins, sigma_x_bins, sigma_y_bins, sigma_yaw_bins, reliable_value_bins = dict(), dict(), dict(), dict(), dict(), dict()
        for i in range(len(AREs)):
            # ratio = np.maximum(AREs[i] / ARE_thresh, ATEs[i] / ATE_thresh)
            # ratio = math.ceil(ratio)
            ratio = reliable_values[i] / reliable_thresh
            count = 0
            while ratio >= 1:
                count += 1
                ratio /= 2
            ratio = count
            if ratio not in sigma_x_bins:
                ARE_bins[ratio] = []
                ATE_bins[ratio] = []
                sigma_x_bins[ratio] = []
                sigma_y_bins[ratio] = []
                sigma_yaw_bins[ratio] = []
                reliable_value_bins[ratio] = []
            ARE_bins[ratio].append(AREs[i])
            ATE_bins[ratio].append(ATEs[i])
            sigma_x_bins[ratio].append(sigma_xs[i])
            sigma_y_bins[ratio].append(sigma_ys[i])
            sigma_yaw_bins[ratio].append(sigma_yaws[i])
            reliable_value_bins[ratio].append(reliable_values[i])
        # mean, std
        ARE_mean, ARE_std = dict(), dict()
        ATE_mean, ATE_std = dict(), dict()
        sigma_x_mean, sigma_x_std = dict(), dict()
        sigma_y_mean, sigma_y_std = dict(), dict()
        sigma_yaw_mean, sigma_yaw_std = dict(), dict()
        reliable_value_mean, reliable_value_std = dict(), dict()
        ratios = np.sort(list(sigma_x_bins.keys()))
        for ratio in ratios:
            self.logger.info(f'#######################{ratio}#######################')
            self.logger.info(f'bin size: {len(sigma_x_bins[ratio])}')
            ARE_mean[ratio] = np.mean(ARE_bins[ratio])
            ARE_std[ratio] = np.std(ARE_bins[ratio])
            ATE_mean[ratio] = np.mean(ATE_bins[ratio])
            ATE_std[ratio] = np.std(ATE_bins[ratio])
            sigma_x_mean[ratio] = np.mean(sigma_x_bins[ratio])
            sigma_x_std[ratio] = np.std(sigma_x_bins[ratio])
            sigma_y_mean[ratio] = np.mean(sigma_y_bins[ratio])
            sigma_y_std[ratio] = np.std(sigma_y_bins[ratio])
            sigma_yaw_mean[ratio] = np.mean(sigma_yaw_bins[ratio])
            sigma_yaw_std[ratio] = np.std(sigma_yaw_bins[ratio])
            reliable_value_mean[ratio] = np.mean(reliable_value_bins[ratio])
            reliable_value_std[ratio] = np.std(reliable_value_bins[ratio])
            self.logger.info(f'ARE_mean: {ARE_mean[ratio]:<.2f}, ARE_std: {ARE_std[ratio]:<.2f}')
            self.logger.info(f'ATE_mean: {ATE_mean[ratio]:<.2f}, ATE_std: {ATE_std[ratio]:<.2f}')
            self.logger.info(f'sigma_x_mean: {sigma_x_mean[ratio]:<.2f}, sigma_x_std: {sigma_x_std[ratio]:<.2f}')
            self.logger.info(f'sigma_y_mean: {sigma_y_mean[ratio]:<.2f}, sigma_y_std: {sigma_y_std[ratio]:<.2f}')
            self.logger.info(f'sigma_yaw_mean: {sigma_yaw_mean[ratio]:<.2f}, sigma_yaw_std: {sigma_yaw_std[ratio]:<.2f}')
            self.logger.info(f'reliable_value_mean: {reliable_value_mean[ratio]:<.5f}, reliable_value_std: {reliable_value_std[ratio]:<.5f}\n')