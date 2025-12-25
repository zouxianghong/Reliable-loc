# Particle localization module
# Created by Ericxhzou

import numpy as np
import os
import time
import matplotlib.pyplot as plt
from queue import Queue

import torch

from monte_carlo_loc.sensor_model import SensorModel
from monte_carlo_loc.motion_model import MotionModel
from monte_carlo_loc.resample import resample
from monte_carlo_loc.sgv_utils import match_pair_parallel
from monte_carlo_loc.visualizer import Visualizer
from monte_carlo_loc.vis_loc_result import plot_traj_result


class PFLoc:
    def __init__(self, logger, config, dataset, poses):
        self.log_dir = config['log_dir']
        self.logger = logger
        self.dataset_name = config['dataset']
        self.dataset = dataset
        self.poses = poses
        self.start_idx = config['start_idx']
        self.end_idx = config['end_idx'] if config['end_idx'] < len(poses) else len(poses)
        self.num_particles = config['num_particles']
        self.move_thres = config['move_thres']
        self.use_sgv = config['use_sgv']
        self.use_yaw = config['use_yaw']
        self.is_initial = False
        # motion model
        self.motion_model = MotionModel(config, poses)
        # sensor model
        self.sensor_model = SensorModel(logger, config, dataset, poses)
        self.sensor_model.construct_map_tree()
        self.particles = self.sensor_model.init_particles(self.num_particles,
                                                          config['init_type'],
                                                          x=poses[self.start_idx, 0, 3],
                                                          y=poses[self.start_idx, 1, 3],
                                                          z=0.0,
                                                          yaw=self.motion_model.gt_headings[self.start_idx])
        # vis
        self.visualizer = None
        # save result
        self.loc_results = np.empty((len(poses), self.num_particles, 5))
        self.num_occupancies = np.zeros(len(poses))
    
    def log_time_cost(self):
        # Particle cluster
        if len(self.sensor_model.time_costs_particle_cluster) > 0:
            mean_time_cost = np.mean(self.sensor_model.time_costs_particle_cluster)
            std_time_cost = np.std(self.sensor_model.time_costs_particle_cluster)
            self.logger.info(f'[Particle Clustering time cost]: {mean_time_cost:<.2f} ± {std_time_cost:<.2f}ms')
        # Global feature scoring
        if len(self.sensor_model.time_costs_global):
            mean_time_cost = np.mean(self.sensor_model.time_costs_global)
            std_time_cost = np.std(self.sensor_model.time_costs_global)
            self.logger.info(f'[Global Feature Scoring time cost]: {mean_time_cost:<.2f} ± {std_time_cost:<.2f}ms')
        # SGV scoring
        if len(self.sensor_model.time_costs_sgv) > 0:
            mean_time_cost = np.mean(self.sensor_model.time_costs_sgv)
            std_time_cost = np.std(self.sensor_model.time_costs_sgv)
            self.logger.info(f'[SGV Scoring time cost]: {mean_time_cost:<.2f} ± {std_time_cost:<.2f}ms')
    
    def run_one_step(self, frame_idx):
        # motion model
        self.particles = self.motion_model.move(self.particles, frame_idx)

        # only update the weight when the car moves
        if self.motion_model.commands[frame_idx, 1] > self.move_thres or self.is_initial:
            self.is_initial = False

            # grid-based method
            if self.use_sgv:
                self.particles = self.sensor_model.update_weights2(self.particles, self.sensor_model.query_indices[frame_idx], use_yaw=self.use_yaw)
            else:
                self.particles = self.sensor_model.update_weights(self.particles, self.sensor_model.query_indices[frame_idx])
            
            # # for debug:
            # if self.particles is not None:
            #     occupancy_states = np.zeros(len(self.sensor_model.map_indices), dtype=np.int32)
            #     for i in range(len(self.particles)):
            #         particle_pos = np.array([[self.particles[i, 0], self.particles[i, 1]]])
            #         dists, indices = self.sensor_model.map_pos_tree.query(particle_pos, k=1)
            #         if dists[0][0] >= 30.0:  # skip if too far
            #             continue
            #         occupancy_states[indices[0][0]] += 1
            #     num_occupancy = len(occupancy_states[occupancy_states > 0])
            #     self.num_occupancies[frame_idx] = num_occupancy

            # resampling
            if self.particles is not None:
                self.particles = resample(self.particles)
            
            # # initialize again if no valid particles
            # if self.particles is None:
            #     self.particles = self.sensor_model.init_particles(self.num_particles)
            #     self.logger.info(f'Initialize particles! frame: {frame_idx}')
    
    def show_sim_graph(self):
        g_offset = self.dataset.data_cfg['global_offset']
        for frame_idx in range(self.start_idx, self.end_idx):
            self.sensor_model.vis_sim_graph(self.poses, frame_idx)