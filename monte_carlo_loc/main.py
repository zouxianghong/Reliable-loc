#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: this is the main script file for overlap-based Monte Carlo localization.

import os
import sys
import argparse
from datetime import datetime
import yaml

from utils.util import check_makedirs
from datasets.place_recognition_dataset import PlaceRecognitionDataSet

from monte_carlo_loc import loc_utils
from monte_carlo_loc.reliable_loc import ReliableLoc

def get_args():
    parser = argparse.ArgumentParser(description='Reliable Point Cloud Localization')
    parser.add_argument('--config', type=str, default='configs/mc_loc.yaml', help='config file')
    parser.add_argument('--dataset', type=str, default=None,
                        help='cs_college, info_campus, zhongshan_park, jiefang_road, yanjiang_road1, yanjiang_road2')
    parser.add_argument('--start_idx', type=int, default=0, help='cs_college: 0, \
                                                                  info_campus: 300, \
                                                                  zhongshan_park: 0, \
                                                                  jiefang_road: 0, \
                                                                  yanjiang_road1: 400, \
                                                                  yanjiang_road2: 0')
    parser.add_argument('--end_idx', type=int, default=10000, help='end frame idx')
    parser.add_argument('--init_type', type=str, default='default', help='default, gt_xy_yaw, gt_xy')
    parser.add_argument('--use_sgv', default=False, action='store_true', help='use spectral geometric verfication to adjust particle weight')
    parser.add_argument('--use_yaw', default=False, action='store_true', help='use yaw estmated by SVD to adjust particle weight')
    parser.add_argument('--min_reliable_value', type=float, default=0.001, help='min reliable value for reg cerification')
    parser.add_argument('--loc_mode', type=int, default=0, help='0: use only particle filter and without any certification\
                                                                -1: use only reg loc once pf loc is converged\
                                                                 1: switch between pf loc and reg loc adaptively')
    parser.add_argument('--test_name', type=str, default=None, help='scoring, param, necessity')
    args = parser.parse_args()
    # load config file
    if yaml.__version__ >= '5.1':
        config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    else:
        config = yaml.load(open(args.config))
    config['dataset'] = args.dataset
    config['start_idx'] = args.start_idx
    config['end_idx'] = args.end_idx
    config['init_type'] = args.init_type
    config['use_sgv'] = args.use_sgv
    config['use_yaw'] = args.use_yaw
    config['min_reliable_value'] = args.min_reliable_value
    config['loc_mode'] = args.loc_mode
    config['test_name'] = args.test_name
    return config


if __name__ == '__main__':
    # params
    config = get_args()
    
    # logger
    dataset_name = config['dataset']
    t_str = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    loc_type = ''
    if config['loc_mode'] == 0:
        loc_type = 'only_pf'
        if config['use_sgv']:
            loc_type = f'{loc_type}_sgv'
            if config['use_yaw']:
                loc_type = f'{loc_type}_yaw'
    elif config['loc_mode'] == 1:
        min_reliable_value = config['min_reliable_value']
        loc_type = f'reliable_{min_reliable_value:<.4f}'
    elif config['loc_mode'] == -1:
        min_reliable_value = config['min_reliable_value']
        loc_type = f'only_reg_{min_reliable_value:<.4f}'
    if config['test_name'] is None:
        config['log_dir'] = os.path.join(config['exp_dir'], f"{t_str}_{dataset_name}_{loc_type}")
    else:
        test_name = config['test_name']
        config['log_dir'] = os.path.join(config['exp_dir'], f"{t_str}_{dataset_name}_{loc_type}_{test_name}")
    check_makedirs(config['log_dir'])
    logger = loc_utils.get_logger(config['log_dir'])
    logger.info(config)

    # load dataset
    dataset = PlaceRecognitionDataSet(name=config['dataset'], for_training=False)
    dataset.dataset.get_indices_in_dataset()

    # load helmet info and poses
    pose_file = os.path.join(dataset.dataset.data_dir(), 'helmet_submap', 'pose.csv')
    poses = loc_utils.load_poses_whu(pose_file)
    g_offset = dataset.dataset.data_cfg['global_offset']
    for i in range(len(poses)):
        poses[i, 0, 3] -= g_offset[0,0]
        poses[i, 1, 3] -= g_offset[0,1]
        poses[i, 2, 3] -= g_offset[0,2]

    # reliable localizer
    loc = ReliableLoc(logger=logger, config=config, dataset=dataset.dataset, poses=poses)
    # loc.run_test(vis_pps=False)
    if config['loc_mode'] == 0:
        loc.run_pf_loc()
    elif config['loc_mode'] == -1:
        loc.run_reg_loc()
    elif config['loc_mode'] == 1:
        loc.run_reliable_loc()