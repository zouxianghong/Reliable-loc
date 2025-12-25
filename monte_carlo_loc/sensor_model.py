#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: this is the sensor model for overlap-based Monte Carlo localization.
#        This model use grid map, where each grid contains a virtual frame.
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from sklearn.metrics.pairwise import cosine_similarity
import torch
import cupy
import time

from libs.pointops.functions import pointops
from monte_carlo_loc import sgv_utils
from monte_carlo_loc.initialization import init_particles_given_coords
from loc_utils import euler_angles_from_rotation_matrix

class SensorModel():
    """ This class is the implementation of using overlap predictions from OverlapNet
        as the sensor model for localization. In this sensor model we discretize the environment and generate a virtual
        frame for each grid after discretization. We use OverlapNet estimate the overlaps between the current frame and
        the grid virtual frames and use the predictions as the observation measurement.
    """
    def __init__(self, logger, config, dataset, poses):
        """ initialization:
            config_file: the configuration file of the OverlapNet
        """
        self.model_type = config['model_type']
        self.dataset = dataset
        self.query_poses = poses
        self.map_indices = None
        self.map_uuid_idx = None
        self.map_positions = None
        self.map_pos_tree = None
        self.map_feat_tree = None
        self.num_particles = config['num_particles']
        # converge
        self.is_converged = False
        self.num_reduced = config['num_reduced']
        self.converge_thres = config['converge_thres']
        self.num_occupancy = 0
        # parameters for weight updating
        self.default_weight = 0.1
        self.invalid_weight = 0.001
        # 
        self.count_invalid = 0
        self.logger = logger
        self.use_sgv = False
        self.all_sims = None
        # time cost
        self.time_costs_particle_cluster = []
        self.time_costs_global = []
        self.time_costs_sgv = []

    def init_particles(self, num_particles, init_type='default', x=None, y=None, z=None, yaw=None, radius=None, cov=None):
        self.is_converged = False
        self.count_invalid = 0
        self.use_sgv = False
        if init_type == 'default':
            particles = init_particles_given_coords(num_particles, self.map_positions)
        elif init_type == 'gt_xy_yaw':
            particles = np.array([[x, y, z, yaw, 1.0]])
            particles = np.repeat(particles, num_particles, axis=0)
        elif init_type == 'gt_xy':
            grid_coords = np.array([[x, y]])
            particles = init_particles_given_coords(num_particles, grid_coords)
        elif init_type == 'radius_yaw':
            particles = []
            for i in range(num_particles):
                x_i = x + radius * np.random.uniform(-1, 1)
                y_i = y + radius * np.random.uniform(-1, 1)
                yaw_i = yaw + np.pi / 2 * np.random.uniform(-1, 1)
                if yaw_i < -np.pi:
                    yaw_i += 2 * np.pi
                elif yaw_i > np.pi:
                    yaw_i -= 2 * np.pi
                particles.append([x_i, y_i, z, yaw_i, 1.0])
            particles = np.array(particles)
        elif init_type == 'cov':
            epsilon = 1e-10
            # ellipse
            sigma_xx = cov[0,0]
            sigma_xy = cov[0,1]
            sigma_yy = cov[1,1]
            part1 = (sigma_xx + sigma_yy) / 2
            part2 = np.sqrt((sigma_xx - sigma_yy)**2 / 4 + sigma_xy**2)
            ellipse_width = np.sqrt(part1 + part2)
            ellipse_height = np.sqrt(part1 - part2)
            ellipse_theta = 0
            if np.abs(sigma_xy) < epsilon:
                if sigma_xx >= sigma_yy:
                    ellipse_theta = 0
                else:
                    ellipse_theta = np.pi / 2
            else:
                ellipse_theta = np.arctan2(ellipse_width**2 - sigma_xx, sigma_xy)
            # particles
            particles, ratio = [], 5.54
            sigma_yaw = np.sqrt(cov[2,2])
            for i in range(num_particles):
                lambda1 = ratio * np.random.uniform(0, 1) * ellipse_width + epsilon
                lambda2 = ratio * np.random.uniform(0, 1) * ellipse_height + epsilon
                t = np.random.uniform(0, 1) * np.pi * 2
                delta_x_i = lambda1*np.cos(ellipse_theta)*np.cos(t) - lambda2*np.sin(ellipse_theta)*np.sin(t)
                x_i = x + delta_x_i
                delta_y_i = lambda1*np.sin(ellipse_theta)*np.cos(t) + lambda2*np.cos(ellipse_theta)*np.sin(t)
                y_i = y + delta_y_i
                delta_yaw_i = ratio * np.random.uniform(-1, 1) * sigma_yaw
                yaw_i = yaw + delta_yaw_i
                if yaw_i < -np.pi:
                    yaw_i += 2 * np.pi
                elif yaw_i > np.pi:
                    yaw_i -= 2 * np.pi
                wieight_x_i = np.exp(-0.5 * delta_x_i * delta_x_i / sigma_xx)
                wieight_y_i = np.exp(-0.5 * delta_y_i * delta_y_i / sigma_yy)
                wieight_yaw_i = np.exp(-0.5 * delta_yaw_i * delta_yaw_i / (sigma_yaw**2))
                weight_i = wieight_x_i * wieight_y_i * wieight_yaw_i
                particles.append([x_i, y_i, z, yaw_i, weight_i])
            particles = np.array(particles)
        else:
            assert 'Fail to init particles!'
        return particles
    
    def get_xy_idx_by_pos(self, pos_xy):
        resolution = 10.0
        x_idx, y_idx = 0, 0
        # x idx
        if pos_xy[0] >= 0.5 * resolution:
            x_idx = int((pos_xy[0] + 0.5 * resolution) / resolution)
        elif pos_xy[0] < -0.5 * resolution:
            x_idx = int((pos_xy[0] - 0.5 * resolution) / resolution)
        # y idx
        if pos_xy[1] >= 0.5 * resolution:
            y_idx = int((pos_xy[1] + 0.5 * resolution) / resolution)
        elif pos_xy[1] < -0.5 * resolution:
            y_idx = int((pos_xy[1] - 0.5 * resolution) / resolution)
        return x_idx, y_idx
        
    def construct_map_tree(self):
        query_trip_idx = self.dataset.get_trip_idx_by_name("helmet_submap")
        self.query_indices = self.dataset.sample_indices[query_trip_idx]
        map_trip_idx = self.dataset.get_trip_idx_by_name("map_submap_in_grid")
        self.map_indices = self.dataset.sample_indices[map_trip_idx]
        # map positions
        self.map_uuid_idx, self.map_positions = dict(), []
        for idx in self.map_indices:
            pos_xy = self.dataset.get_pos_xy(idx)
            x_idx, y_idx = self.get_xy_idx_by_pos(pos_xy)
            self.map_uuid_idx[x_idx, y_idx] = idx
            self.map_positions.append(pos_xy)
        self.map_positions = np.stack(self.map_positions, axis=0)
        self.map_pos_tree = KDTree(self.map_positions)
        # map features
        map_feats = self.dataset.get_g_descs(self.model_type, self.map_indices)
        map_feats = np.concatenate(map_feats, axis=0)
        self.map_feat_tree = KDTree(map_feats)
    
    def get_map_size(self):
        grid_coords = self.map_positions
        map_size = [np.min(grid_coords[:, 0]), np.max(grid_coords[:, 0]),
                    np.min(grid_coords[:, 1]), np.max(grid_coords[:, 1])]
        return map_size
    
    def cluster_particles(self, particles, cluster_dist_thresh=5.0, max_dist=30.0):
        t0 = time.time()
        cluster_dist_thresh = cluster_dist_thresh**2
        clusters = []
        init_cluster = {
            'idxs': [0],
            'center': np.array([particles[0, 0], particles[0, 1]])
        }
        clusters.append(init_cluster)
        for i in range(1, len(particles)):
            x, y = particles[i, 0], particles[i, 1]
            create_new_cluster = True
            for j in range(len(clusters)):
                in_cluster = False
                center_x, center_y = clusters[j]['center'][0], clusters[j]['center'][1]
                dist = (x-center_x)**2 + (y-center_y)**2
                if dist < cluster_dist_thresh:
                    in_cluster = True
                    clusters[j]['idxs'].append(i)
                    # update cluster center
                    clusters[j]['center'][0] = (center_x * (len(clusters[j]['idxs']) - 1) + x) / len(clusters[j]['idxs'])
                    clusters[j]['center'][1] = (center_y * (len(clusters[j]['idxs']) - 1) + y) / len(clusters[j]['idxs'])
                if in_cluster:
                    create_new_cluster = False
                    break
            if create_new_cluster:
                new_cluster = {
                    'idxs': [i],
                    'center': np.array([particles[i, 0], particles[i, 1]])
                }
                clusters.append(new_cluster)
        # nearest map grid
        k = 5
        for i in range(len(clusters)):
            center = clusters[i]['center']
            dists, indices = self.map_pos_tree.query(center.reshape(1, -1), k=k)
            clusters[i]['map_grid_idx'] = []
            for j in range(len(indices[0])):
                if dists[0][j] > max_dist:
                    break
                clusters[i]['map_grid_idx'].append(self.map_indices[indices[0][j]])
        t1 = time.time()
        time_cost_particle_cluster = (t1 - t0) * 1000  # unit: ms
        self.time_costs_particle_cluster.append(time_cost_particle_cluster)
        return clusters
    
    def update_weights_by_global_features(self, particles, frame_idx, max_dist=30.0):
        t0 = time.time()
        # record which one is sampled already and its index
        overlap_lut = np.ones(len(self.map_indices), dtype=np.int32) * -1
        new_particles = particles
        infer_map_indices = []
        overlap_idxes = 0

        # first collect the grid indexes to calculate overlaps
        query_feat = self.dataset.get_g_desc(self.model_type, frame_idx)
        if query_feat is None:
            return None
        self.all_sims = np.ones(len(particles)) * self.default_weight
        for idx in range(len(particles)):
            particle = particles[idx]
            if particles[idx, -2] < -np.pi:
                particles[idx, -2] += 2 * np.pi
            elif particles[idx, -2] > np.pi:
                particles[idx, -2] -= 2 * np.pi
            # get the nearest map submap
            particle_pos = np.array([[particle[0], particle[1]]])
            dists, indices = self.map_pos_tree.query(particle_pos, k=1)
            if dists[0][0] >= max_dist:  # skip if too far
                self.all_sims[idx] = self.invalid_weight
                continue
            # skip if the feature doesn't exist
            dists, indices = list(dists[0]), list(indices[0])
            valid_dists, valid_indices = [], []
            for i in range(len(indices)):
                if dists[i] > max_dist:
                    continue
                map_feat = self.dataset.get_g_desc(self.model_type, indices[i] + self.map_indices[0])
                if map_feat is None:
                    continue
                valid_dists.append(dists[i])
                valid_indices.append(indices[i])
            if len(valid_dists) < 1:
                continue
            # check a map sub-map sampled or not
            if overlap_lut[valid_indices[0]] < 0:  # if have not been sampled
                overlap_idxes += 1
                infer_map_indices.append(valid_indices[0])
                overlap_lut[valid_indices[0]] = overlap_idxes
            # infer sims
            map_feat = self.dataset.get_g_desc(self.model_type, valid_indices[0] + self.map_indices[0])
            sim = (cosine_similarity(query_feat, map_feat)[0,0] + 1) / 2
            self.all_sims[idx] = sim

        # if no new inferring, skip the weight updating
        if len(infer_map_indices) == 0:
            return None

        # for debug: save real time similarity
        save_real_time_sim = False
        if save_real_time_sim:
            import csv
            from utils.util import check_makedirs
            sim_dir = '/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/sim'
            check_makedirs(sim_dir)
            real_time_sims = np.concatenate((new_particles[:, :2], self.all_sims.reshape(-1, 1)), axis=-1)
            sim_csv = os.path.join(sim_dir, f'sims_{frame_idx}.csv')
            with open(sim_csv, 'w', newline='', encoding='utf-8') as file:
                mywriter = csv.writer(file, delimiter=',')
                mywriter.writerows(real_time_sims)
        
        # update the weights of the particles
        new_particles[:, -1] = new_particles[:, -1] * self.all_sims
        
        t1 = time.time()
        time_cost_global = (t1 - t0) * 1000  # unit: ms
        self.time_costs_global.append(time_cost_global)
        return new_particles
    
    def get_near_kpts(self, kpts, feats, center, num_sample=512, radius=21.2):
        """ Input Tensors: kpts: b x k x 3, feats: b x k x d, center: b x 3
            Return Tensors: near_kpts: b x s x 3, near_feats: b x s x d
        """
        kpts, feats, center = kpts.cuda(), feats.cuda(), center.cuda()
        # sample indices
        kpts_xyz = torch.zeros_like(kpts)
        kpts_xyz[:, :, :2] = kpts[:, :, :2]
        center_xyz = torch.zeros_like(center)
        center_xyz[:, :2] = center[:, :2]
        center_xyz = center_xyz[:, None, :]
        kpts_xyz = kpts_xyz - center_xyz
        center_xyz = torch.zeros_like(center_xyz)
        kpts_xyz, center_xyz = kpts_xyz.contiguous().float(), center_xyz.contiguous().float()
        idx = pointops.ballquery(radius, num_sample, kpts_xyz, center_xyz).squeeze(1)  # b x s
        # kpts / feats
        kpts = kpts.transpose(1, 2).contiguous().float()  # b x 3 x k
        near_kpts = pointops.gathering(kpts, idx).transpose(1, 2).contiguous()  # b x s x 3
        feats = feats.transpose(1, 2).contiguous().float()  # b x d x k
        near_feats = pointops.gathering(feats, idx).transpose(1, 2).contiguous()  # b x s x d
        return near_kpts, near_feats
    
    def adjust_weights_by_sgv(self, clusters, particles, frame_idx, use_yaw = False):
        t0 = time.time()
        # check query feats
        q_pos_xyz = self.query_poses[frame_idx, :3, 3]
        q_l_kpt, q_l_feat = self.dataset.get_l_kpt_desc(self.model_type, frame_idx, unify_coord=True, g_offset=False)
        if q_l_kpt is None or q_l_feat is None:
            return particles
        # check particle clusters and map features
        clusters_are_valid, m_kpts, m_feats, m_centers = np.zeros(len(clusters), dtype=bool), [], [], []
        for i in range(len(clusters)):
            m_l_kpt, m_l_feat = None, None
            for map_grid_idx in clusters[i]['map_grid_idx']:
                if map_grid_idx == -1:
                    break
                m_l_kpt, m_l_feat = self.dataset.get_l_kpt_desc(self.model_type, map_grid_idx, unify_coord=True, g_offset=False)
                if m_l_kpt is None or m_l_feat is None:  # no feat, should not happen!
                    continue
            if m_l_kpt is None or m_l_feat is None:  # no feat, should not happen!
                continue
            clusters_are_valid[i] = True
            m_kpts.append(m_l_kpt)
            m_feats.append(m_l_feat)
            m_center = self.dataset.get_pos_xyz(map_grid_idx, with_offset=False)
            m_centers.append(m_center)
        if len(m_kpts) == 0:  # skip if no valid clusters or no valid features
            return particles
        # effective trick: avoid the disturbance of further context when particles gather together
        use_trick_near_kpts = False
        if use_trick_near_kpts:
            num_sample, radius = 512, 21.2
            q_center = self.dataset.get_pos_xyz(frame_idx, with_offset=False)
            q_kpts_centered = (q_l_kpt - q_center.reshape(1, -1))[:, :2]  # k x 2
            q_dist = np.linalg.norm(q_kpts_centered, axis=-1)
            use_trick_near_kpts = np.sum(q_dist < radius) >= num_sample and 1 < len(clusters) < 5
        # query tensors
        q_kpts = torch.from_numpy(q_l_kpt[None, ...])  # 1 x k x 3
        q_feats = torch.from_numpy(q_l_feat[None, ...])  # 1 x k x d
        if use_trick_near_kpts:
            q_center = torch.from_numpy(q_center).view(1, -1)  # 1 x 3
            q_kpts, q_feats = self.get_near_kpts(q_kpts, q_feats, q_center, num_sample, radius)  # 1 x s x *
            q_kpts, q_feats = q_kpts.double(), q_feats.double()
        q_kpts, q_feats = q_kpts.repeat(len(m_kpts), 1, 1), q_feats.repeat(len(m_kpts), 1, 1)  # b x s x *
        # map tensors
        m_kpts = torch.from_numpy(np.stack(m_kpts, axis=0))  # b x k x 3
        m_feats = torch.from_numpy(np.stack(m_feats, axis=0))  # b x k x d
        if use_trick_near_kpts:
            m_centers = torch.from_numpy(np.stack(m_centers, axis=0))  # b x 3
            m_kpts, m_feats = self.get_near_kpts(m_kpts, m_feats, m_centers, num_sample, radius)  # b x s x *
            m_kpts, m_feats = m_kpts.double(), m_feats.double()
        # parallel spectral geometric verification
        sgv_scores, q_kpts, m_kpts, qm_weights = sgv_utils.sgv_fn(q_feats, q_kpts, m_feats, m_kpts, d_thresh=5.0, use_cuda=True)
        sgv_scores = sgv_scores.reshape(1) if len(sgv_scores.shape) == 0 else sgv_scores
        min_score, max_score = np.min(sgv_scores), np.max(sgv_scores)
        sgv_scores = sgv_scores / max_score
        self.logger.info(f'[Adjust weights by SGV] frame: {frame_idx} num cluster: {len(clusters)}, scores" {min_score:<.4f}~{max_score:<.4f}, use near: {use_trick_near_kpts}')
        
        # for debug: vis pps
        vis_pps = False  # frame_idx > 1950
        if vis_pps:
            from utils.draw_result import draw_pps
            draw_pps(q_kpts[0], m_kpts[0])
        
        # SVD
        if use_yaw:
            # refer to: https://github.com/qinzheng93/GeoTransformer/blob/main/geotransformer/modules/registration/procrustes.py
            # weights normalization
            weight_thresh, eps = 0.75, 1.e-5
            qm_weights[qm_weights < weight_thresh] = 0.0
            qm_weights = qm_weights / (np.sum(qm_weights, axis=1, keepdims=True) + eps)  # b x k
            qm_weights = qm_weights[..., None]  # b x k x 1
            q_centers = np.sum(q_kpts * qm_weights, axis=1, keepdims=True)  # b x 1 x 3
            m_centers = np.sum(m_kpts * qm_weights, axis=1, keepdims=True)  # b x 1 x 3
            q_kpts_centered = q_kpts - q_centers  # b x k x 3
            m_kpts_centered = m_kpts - m_centers  # b x k x 3
            m = np.matmul(m_kpts_centered.transpose(0, 2, 1), qm_weights * q_kpts_centered)  # b x 3 x 3
            u, s, v = cupy.linalg.svd(cupy.asarray(m))  # b x 3 x 3, b x 3, b x 3 x 3
            R = cupy.asnumpy(cupy.matmul(v.transpose(0, 2, 1), u.transpose(0, 2, 1)))  # predicted RT, b x 3 x 3
            t = q_centers - np.matmul(m_centers, R.transpose(0, 2, 1))  # b x 1 x 3
            # estimate query center xyz by R & t
            xyzs = []
            yaws = []
            for i in range(q_kpts.shape[0]):
                xyz = (R[i] @ (q_pos_xyz.reshape(-1,1))).reshape(-1) + t[i]
                xyzs.append(xyz[0])
                phi = euler_angles_from_rotation_matrix(R[i])[2]
                yaws.append(phi)
        
        # adjust weights
        count, xy_sigma, yaw_sigma = 0, 30, 60.0 * np.pi / 180
        for i in range(len(clusters)):
            if clusters_are_valid[i]:
                sgv_score = sgv_scores[count]
                if use_yaw:
                    sgv_xy = xyzs[count][:2]
                    sgv_yaw = yaws[count]
                count += 1
            else:
                sgv_score = min_score / max_score
                sgv_xy = None
                sgv_yaw = None
            idxs = clusters[i]['idxs']
            for idx in idxs:
                particles[idx, -1] = particles[idx, -1] * sgv_score
                if use_yaw and self.is_converged:
                    if sgv_yaw is None:  # self.all_sims[idx] < 0.75
                        xy_score = 0.1
                        yaw_score = 0.1
                    else:
                        delta_xy_dist = np.linalg.norm(sgv_xy - particles[idx, :2])
                        xy_score = np.exp(-0.5 * delta_xy_dist * delta_xy_dist / (xy_sigma * xy_sigma))
                        xy_score = np.maximum(xy_score, self.invalid_weight)
                        delta_yaw = min(abs(sgv_yaw - particles[idx, -2]), 2*np.pi - abs(sgv_yaw - particles[idx, -2]))
                        yaw_score = np.exp(-0.5 * delta_yaw * delta_yaw / (yaw_sigma * yaw_sigma))
                        yaw_score = np.maximum(yaw_score, self.invalid_weight)
                    particles[idx, -1] = particles[idx, -1] * xy_score * yaw_score
        t1 = time.time()
        time_cost_sgv = (t1 - t0) * 1000  # unit: ms
        self.time_costs_sgv.append(time_cost_sgv)
        return particles
    
    def check_convergency(self, particles, max_dist=30.0):
        if particles is None:
            return None
        occupancy_states = np.zeros(len(self.map_indices), dtype=np.int32)
        for i in range(len(particles)):
            particle_pos = np.array([[particles[i, 0], particles[i, 1]]])
            dists, indices = self.map_pos_tree.query(particle_pos, k=1)
            if dists[0][0] >= max_dist:  # skip if too far
                continue
            occupancy_states[indices[0][0]] += 1
        self.num_occupancy = len(occupancy_states[occupancy_states > 0])
        if self.num_occupancy == 0:
            self.count_invalid += 1
        if self.count_invalid == 10:  # no valid particles in base map, initialize again!
            self.count_invalid = 0
            return None
        if 0 < self.num_occupancy < self.converge_thres and not self.is_converged:
            self.is_converged = True
            idxes = np.argsort(particles[:, -1])[::-1]
            if not self.num_reduced > len(particles) or self.num_reduced < 0:
                particles = particles[idxes[:self.num_reduced]]
        return particles
    
    def normalize_particles(self, particles):
        if particles is not None:
            particles[:, -1] = particles[:, -1] / np.max(particles[:, -1])
        return particles
    
    def vis_sim_graph(self, poses, frame_idx):
        import csv
        from utils.util import check_makedirs
        sim_dir = '/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/sim'
        check_makedirs(sim_dir)
        query_feat = self.dataset.get_g_desc(self.model_type, frame_idx)
        query_xy = np.array([poses[frame_idx, 0, 3], poses[frame_idx, 1, 3]])
        if query_feat is None:
            return
        xysim = []
        xysim.append(np.array([query_xy[0], query_xy[1], 0.2]))
        for i in range(len(self.map_indices)):
            map_xy = self.map_positions[i]
            dist = np.linalg.norm(query_xy - map_xy)
            if dist > 100.0:
                continue
            map_feat = self.dataset.get_g_desc(self.model_type, self.map_indices[i])
            if map_feat is None:
                continue
            sim = (cosine_similarity(query_feat, map_feat)[0,0] + 1) / 2
            xysim.append(np.array([map_xy[0], map_xy[1], sim]))
        xysim = np.stack(xysim, axis=0)
        sim_csv = os.path.join(sim_dir, f'sims_{frame_idx}.csv')
        with open(sim_csv, 'w', newline='', encoding='utf-8') as file:
            mywriter = csv.writer(file, delimiter=',')
            mywriter.writerows(xysim)

    def update_weights2(self, particles, frame_idx, max_dist=30.0, use_yaw=False):
        # update weight by global features
        self.num_occupancy = 0
        particles = self.update_weights_by_global_features(particles, frame_idx, max_dist)
        if particles is None:
            return None
        # cluster particles
        max_num_cluster = 40
        clusters = self.cluster_particles(particles, cluster_dist_thresh=5.0, max_dist=max_dist)
        if not self.use_sgv and len(clusters) <= max_num_cluster:
            self.use_sgv = True
        if not self.use_sgv or len(clusters) == 0:
            particles = self.normalize_particles(particles)
            return particles
        # adjust particles weights by sgv
        particles = self.adjust_weights_by_sgv(clusters, particles, frame_idx, use_yaw)
        # check convergence using the number of occupied grids
        last_state = self.is_converged
        particles = self.check_convergency(particles, max_dist)
        if not last_state and self.is_converged:
            self.logger.info(f'[Check Convergency] Converged! frame idx: {frame_idx}')
        # normalization
        particles = self.normalize_particles(particles)

        return particles
    
    def update_weights(self, particles, frame_idx, max_dist=30.0):
        """ This function update the weight for each particle using batch.
            Args:
                particles: each particle has four properties [x, y, theta, weight]
            Returns:
                particles ... same particles with changed particles(i).weight
        """
        # update particles' weights by global features
        self.num_occupancy = 0
        new_particles = self.update_weights_by_global_features(particles, frame_idx, max_dist)

        # check convergence using the number of occupied grids
        last_state = self.is_converged
        new_particles = self.check_convergency(new_particles, max_dist)
        if not last_state and self.is_converged:
            self.logger.info(f'[Check Convergency] Converged! frame idx: {frame_idx}')

        # normalization
        new_particles = self.normalize_particles(new_particles)

        return new_particles
