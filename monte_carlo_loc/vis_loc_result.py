#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: This script plots the final localization results and can visualize offline given the results.

import os
import sys
import argparse
import yaml
import loc_utils
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')

from utils.util import check_makedirs, get_sub_dirs
from datasets.place_recognition_dataset import PlaceRecognitionDataSet
from motion_model import MotionModel
from sensor_model import SensorModel
from visualizer import Visualizer


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
    return config


def plot_traj_result(results, poses, gt_headings, numParticles=1000, start_idx=0, end_idx=0,
                     ratio=0.25, save_filepath=None, logger=None):
    """ Plot the final localization trajectory.
        Args:
            results: localization results including particles in every timestamp.
            poses: ground truth poses.
            numParticles: number of particles.
            start_idx: the start index.
            ratio: the ratio of particles used to estimate the poes.
            converge_thres: a threshold used to tell whether the localization converged or not.
            eva_thres: a threshold to check the estimation results.
    """
    # get ground truth xy and yaw separately
    gt_location = poses[start_idx:end_idx, :2, 3]
    gt_headings = gt_headings[start_idx:end_idx]

    estimated_traj = []
    for frame_idx in range(start_idx, end_idx):
        particles = results['particles'][frame_idx]
        # collect top 25% of particles to estimate pose
        idxes = np.argsort(particles[:, -1])[::-1]
        idxes = idxes[:int(ratio * numParticles)]
        partial_particles = particles[idxes]
        if np.sum(partial_particles[:, -1]) == 0:
            continue
        normalized_weight = partial_particles[:, -1] / np.sum(partial_particles[:, -1])
        estimated_traj.append(partial_particles[:, :-1].T.dot(normalized_weight.T))
    estimated_traj = np.array(estimated_traj)
    
    # output est pose
    out_file = open('/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/xyzyaw_est.txt', 'w')
    for i in range(len(estimated_traj)):
        loc_mode = results['loc_modes'][i]
        x, y, z, yaw = estimated_traj[i,0], estimated_traj[i,1], estimated_traj[i,2], estimated_traj[i,3]
        out_file.write(f'{loc_mode}, {x}, {y}, {z}, {yaw}\n')
    out_file.close()
    
    # output gt pose
    out_file = open('/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/xyzyaw_gt.txt', 'w')
    gt_poses = poses[start_idx:end_idx, :3, 3]
    for i in range(len(gt_poses)):
        loc_mode = results['loc_modes'][i]
        x, y, z, yaw = gt_poses[i,0], gt_poses[i,1], gt_poses[i,2], gt_headings[i]
        out_file.write(f'{loc_mode}, {x}, {y}, {z}, {yaw}\n')
    out_file.close()
    
    # z errors
    diffs_z = np.abs(np.array(estimated_traj[:, 2] - poses[start_idx:end_idx, 2, 3]))
    diffs_z = diffs_z[results['converge_states']]
    mean_z_err = np.mean(diffs_z)
    mean_square_z_error = np.mean(diffs_z * diffs_z)
    rmse_z_err = np.sqrt(mean_square_z_error)
    logger.info(f'[Z Error]: {mean_z_err:<.2f} ± {rmse_z_err:<.2f}m')

    # evaluate the results
    diffs_seperate = np.array(estimated_traj[:, :2] - gt_location)
    diffs_location = np.linalg.norm(diffs_seperate, axis=1)  # diff in euclidean, unit: m
    diffs_yaw = estimated_traj[:, 3] - gt_headings  # diff of yaw
    diffs_yaw = np.minimum(np.abs(diffs_yaw), 2*np.pi - np.abs(diffs_yaw)) * 180/np.pi  # diff of yaw, unit: deg
    
    # success ratio: Δxy < 2m and Δyaw < 5deg
    suc_ratio = np.sum((diffs_location < 2) & (diffs_yaw < 5)) / len(diffs_location) * 100
    logger.info(f'[Δxy < 2m, Δyaw < 5deg] success ratio: {suc_ratio:<.2f}%')
    
    # success ratio for xy
    logger.info('[xy/yaw] success ratio: <2 | 5 | 10 | 15 | 20 | 25 | 30 m/deg')
    success_ratio_xy = []
    for max_err in [2, 5, 10, 15, 20, 25, 30]:
        diffs_i = diffs_location[diffs_location < max_err]
        suc_ratio = len(diffs_i) / len(diffs_location) * 100
        success_ratio_xy.append(suc_ratio)
    logger.info(f'[xy] {success_ratio_xy[0]:<.2f} {success_ratio_xy[1]:<.2f} {success_ratio_xy[2]:<.2f} {success_ratio_xy[3]:<.2f} {success_ratio_xy[4]:<.2f} {success_ratio_xy[5]:<.2f} {success_ratio_xy[6]:<.2f}')
    
    # success ratio for yaw
    success_ratio_yaw = []
    for max_err in [2, 5, 10, 15, 20, 25, 30]:
        diffs_i = diffs_yaw[diffs_yaw < max_err]
        suc_ratio = len(diffs_i) / len(diffs_yaw) * 100
        success_ratio_yaw.append(suc_ratio)
    logger.info(f'[yaw] {success_ratio_yaw[0]:<.2f} {success_ratio_yaw[1]:<.2f} {success_ratio_yaw[2]:<.2f} {success_ratio_yaw[3]:<.2f} {success_ratio_yaw[4]:<.2f} {success_ratio_yaw[5]:<.2f} {success_ratio_yaw[6]:<.2f}')

    # location error
    converged_diffs_location = diffs_location[results['converge_states']]
    mean_location_err = np.mean(converged_diffs_location)
    mean_square_location_error = np.mean(converged_diffs_location * converged_diffs_location)
    rmse_location_err = np.sqrt(mean_square_location_error)
    logger.info(f'[Location Error]: {mean_location_err:<.2f} ± {rmse_location_err:<.2f}m')

    # yaw error
    converged_diffs_yaw = diffs_yaw[results['converge_states']]
    mean_yaw_err = np.mean(converged_diffs_yaw)
    mean_square_yaw_error = np.mean(converged_diffs_yaw * converged_diffs_yaw)
    rmse_yaw_err = np.sqrt(mean_square_yaw_error)
    logger.info(f'[Yaw Error]: {mean_yaw_err:<.2f} ± {rmse_yaw_err:<.2f}deg')

    # plot results
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111)

    ax.plot(poses[start_idx:end_idx, 0, 3], poses[start_idx:end_idx, 1, 3], c='r', label='ground_truth')
    
    # filter estimated traj far from gt
    draw_traj_list = [[]]
    for i in range(len(estimated_traj)):
        if diffs_location[i] < 100.0:
            draw_traj_list[-1].append(estimated_traj[i])
        else:
            draw_traj_list.append([])
    for i in range(len(draw_traj_list)):
        if len(draw_traj_list[i]) == 0:
            continue
        draw_traj = np.stack(draw_traj_list[i], axis=0)
        ax.plot(draw_traj[:, 0], draw_traj[:, 1], c='b', label='weighted_mean_25%')
    if save_filepath:
        fig.savefig(save_filepath, transparent=False, bbox_inches='tight')
    plt.show()


def vis_offline(results, poses, gt_headings, map_size, numParticles=1000, start_idx=0):
    """ Visualize localization results offline.
        Args:
            results: localization results including particles in every timestamp.
            poses: ground truth poses.
            map_size: size of the map.
            numParticles: number of particles.
            start_idx: the start index.
    """
    plt.ion()
    visualizer = Visualizer(map_size, poses, gt_headings,
                            numParticles=numParticles,
                            start_idx=start_idx)
    
    for frame_idx in range(start_idx, len(poses)):
        particles = results[frame_idx]
        particle_xyc = visualizer.compute_errs_pf(frame_idx, particles)
        visualizer.update(frame_idx, particle_xyc)
        visualizer.fig.canvas.draw()
        visualizer.fig.canvas.flush_events()
    
    # compute recall
    recall = visualizer.compute_recall(sensor_model.map_pos_tree, pos_dist_thresh=15.0)
    print(f'Recall@top1: {recall[0]}%')


def draw_traj_colored_by_loc_mode(is_reg_loc, poses, title='', font_size=15, save_filepath=None):
    # draw traj
    fig = plt.figure()
    fig.patch.set_alpha(0)
    ax = fig.add_subplot(111)
    ax.patch.set_alpha(0)
    reg_pts, pf_pts = [], []
    for i in range(len(is_reg_loc)):
        if i >= len(poses):
            break
        p = poses[i, :2, 3]
        if is_reg_loc[i]:
            reg_pts.append(p)
        else:
            pf_pts.append(p)
    reg_pts = np.array(reg_pts)
    pf_pts = np.array(pf_pts)
    if len(reg_pts) > 0:
        ax.scatter(reg_pts[:, 0], reg_pts[:, 1], s=2, color='red')
    if len(pf_pts) > 0:
        ax.scatter(pf_pts[:, 0], pf_pts[:, 1], s=2, color='green')
    ax.axis()
    ax.set_aspect('equal')
    plt.xticks(fontsize=font_size, fontweight='bold')  # 默认字体大小为10
    plt.yticks(fontsize=font_size, fontweight='bold')
    plt.title(label=title, pad=20, fontsize=font_size, fontweight='bold')  # 默认字体大小为12
    plt.xlabel(xlabel='x', fontsize=font_size, fontweight='bold')
    plt.ylabel(ylabel='y', fontsize=font_size, fontweight='bold')
    # save
    if save_filepath is not None:
        plt.savefig(save_filepath, dpi=200, bbox_inches='tight')
        plt.close('all')
    else:
        plt.show()


class ExpResult:
    def __init__(self) -> None:
        self.dataset_name = None
        self.start_idx = None
        self.end_idx = None
        self.loc_mode = None
        self.use_sgv = None
        self.use_yaw = None
        self.min_reliable_value = None
        self.success_ratio_xy = []
        self.success_ratio_yaw = []
        self.success_ratio_2m_5deg = None
        self.reg_mode_ratio = None
        self.loc_modes = []
        self.position_errs = []
        self.yaw_errs = []
        self.time_cost = []
        self.gt_locations = []
        self.est_locations = []
        self.num_occupancies = None
        self.particles = []
        self.test_name = None  # 'necessity' or 'scoring' or 'param'


class ChartContent:
    def __init__(self) -> None:
        self.name = 'unknown'
        self.fig_color = 'blue'
        self.fig_marker = 'v'
        self.data = []


# draw curve
def draw_line_chart(data_list, title='', xlabel='', ylabel='', xrange=[0, 30],
                    yrange=[50, 100], ystep=10, font_size=15, save_filepath=None):
    x = np.array([2, 5, 10, 15, 20, 25, 30])
    x = x[:len(data_list[0].data)]
    y = np.arange(yrange[0], yrange[1] + ystep, step=ystep)

    # label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
    # color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
    # 线型：- -- -. : ,# marker：. , o v < * + 1
    plt.figure(figsize=(10, 8))
    plt.grid(linestyle="--")  # 设置背景网格线为虚线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['right'].set_visible(False)  # 去掉右边框

    for data_i in data_list:
        plt.plot(x, data_i.data, marker=data_i.fig_marker, color=data_i.fig_color, label=data_i.name, linewidth=1.5)

    plt.xticks(x, fontsize=font_size, fontweight='bold')  # 默认字体大小为10
    plt.yticks(y, fontsize=font_size, fontweight='bold')
    plt.title(label=title, pad=20, fontsize=font_size, fontweight='bold')  # 默认字体大小为12
    plt.xlabel(xlabel=xlabel, fontsize=font_size, fontweight='bold')
    plt.ylabel(ylabel=ylabel, fontsize=font_size, fontweight='bold')
    plt.xlim(xrange[0], xrange[1])  # 设置x轴的范围
    plt.ylim(yrange[0], yrange[1])

    plt.legend()  # 显示各曲线的图例
    plt.legend(loc=4, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=font_size, fontweight='bold')  # 设置图例字体的大小和粗细

    if save_filepath:
        plt.savefig(save_filepath, format='svg') # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中plt.show()
        plt.close('all')
    else:
        plt.show()


def draw_traj(data_list, title, font_size=15, save_filepath=None, transparent=False):
    # draw traj
    fig = plt.figure()
    if transparent:
        fig.patch.set_alpha(0)
    ax = fig.add_subplot(111)
    if transparent:
        ax.patch.set_alpha(0)
    for data_i in data_list:
        if len(data_i.data) > 0:
            ax.plot(data_i.data[:, 0], data_i.data[:, 1], color=data_i.fig_color, label=data_i.name)  # , linewidth=0.5
    ax.axis()
    ax.set_aspect('equal')
    plt.xticks(fontsize=font_size, fontweight='bold')  # 默认字体大小为10
    plt.yticks(fontsize=font_size, fontweight='bold')
    plt.title(label=title, pad=20, fontsize=font_size, fontweight='bold')  # 默认字体大小为12
    plt.xlabel(xlabel='x', fontsize=font_size, fontweight='bold')
    plt.ylabel(ylabel='y', fontsize=font_size, fontweight='bold')
    plt.legend(numpoints=1)  # 显示各曲线的图例
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=font_size, fontweight='bold')  # 设置图例字体的大小和粗细, fontsize=font_size
    # save
    if save_filepath is not None:
        plt.savefig(save_filepath, format='svg')
        plt.close('all')
    else:
        plt.show()


def get_dataset_name(dataset_name):
    if dataset_name == 'cs_college':
        return 'CS college'
    elif dataset_name == 'info_campus':
        return 'Info campus'
    elif dataset_name == 'zhongshan_park':
        return 'Zhongshan park'
    elif dataset_name == 'jiefang_road':
        return 'Jiefang road'
    elif dataset_name == 'yanjiang_road1':
        return 'Yanjiang road 1'
    elif dataset_name == 'yanjiang_road2':
        return 'Yanjiang road 2'
    else:
        return 'Unknown'


def draw_quantitative_eval(results, dataset_name, min_reliable_value, is_xy, fig_dir):
    # draw success ratio
    data_list = []
    min_suc_ratio = 100
    ## PF
    data_pf = ChartContent()
    data_pf.name = 'PF-loc'
    data_pf.fig_color = 'blue'
    data_pf.fig_marker = 'v'
    for res in results:
        if res.test_name == 'scoring' or res.test_name == 'necessity':
            continue
        if res.dataset_name == dataset_name and res.loc_mode == 0 and res.use_sgv == False:
            if is_xy:
                data_pf.data = np.array(res.success_ratio_xy)
            else:
                data_pf.data = np.array(res.success_ratio_yaw)
            break
    min_suc_ratio = np.minimum(min_suc_ratio, data_pf.data[0])
    data_list.append(data_pf)
    ## PF-SGV
    data_pf_sgv = ChartContent()
    data_pf_sgv.name = 'PF-SGV-loc'
    data_pf_sgv.fig_color = 'green'
    data_pf_sgv.fig_marker = 's'
    for res in results:
        if res.test_name == 'scoring' or res.test_name == 'necessity':
            continue
        if res.dataset_name == dataset_name and res.loc_mode == 0 and res.use_sgv == True and res.use_yaw == False:
            if is_xy:
                data_pf_sgv.data = np.array(res.success_ratio_xy)
            else:
                data_pf_sgv.data = np.array(res.success_ratio_yaw)
            break
    min_suc_ratio = np.minimum(min_suc_ratio, data_pf_sgv.data[0])
    data_list.append(data_pf_sgv)
    ## PF-SGV2
    data_pf_sgv2 = ChartContent()
    data_pf_sgv2.name = 'PF-SGV2-loc'
    data_pf_sgv2.fig_color = 'chocolate'
    data_pf_sgv2.fig_marker = 'x'
    for res in results:
        if res.test_name == 'scoring' or res.test_name == 'necessity':
            continue
        if res.dataset_name == dataset_name and res.loc_mode == 0 and res.use_sgv == True and res.use_yaw == True:
            if is_xy:
                data_pf_sgv2.data = np.array(res.success_ratio_xy)
            else:
                data_pf_sgv2.data = np.array(res.success_ratio_yaw)
            break
    min_suc_ratio = np.minimum(min_suc_ratio, data_pf_sgv2.data[0])
    data_list.append(data_pf_sgv2)
    ## Reg
    data_reg = ChartContent()
    data_reg.name = 'Reg-loc'
    data_reg.fig_color = 'cyan'
    data_reg.fig_marker = '^'
    for res in results:
        if res.test_name == 'scoring' or res.test_name == 'necessity':
            continue
        if res.dataset_name == dataset_name and res.loc_mode == -1 and res.use_sgv == True and res.use_yaw == True and np.abs(res.min_reliable_value - min_reliable_value) < 1e-10:
            if is_xy:
                data_reg.data = np.array(res.success_ratio_xy)
            else:
                data_reg.data = np.array(res.success_ratio_yaw)
            break
    min_suc_ratio = np.minimum(min_suc_ratio, data_reg.data[0])
    data_list.append(data_reg)
    ## Reliable
    data_reliable = ChartContent()
    data_reliable.name = 'Reliable-loc (ours)'
    data_reliable.fig_color = 'red'
    data_reliable.fig_marker = 'd'
    for res in results:
        if res.test_name == 'scoring' or res.test_name == 'necessity':
            continue
        if res.dataset_name == dataset_name and res.loc_mode == 1 and res.use_sgv == True and res.use_yaw == True and np.abs(res.min_reliable_value - min_reliable_value) < 1e-10:
            if is_xy:
                data_reliable.data = np.array(res.success_ratio_xy)
            else:
                data_reliable.data = np.array(res.success_ratio_yaw)
            break
    min_suc_ratio = np.minimum(min_suc_ratio, data_reliable.data[0])
    data_list.append(data_reliable)
    min_suc_ratio = np.floor(min_suc_ratio / 10) * 10
    type_str = 'xy' if is_xy else 'yaw'
    save_filepath = os.path.join(fig_dir, f'quantitative_eval_{dataset_name}_{type_str}.svg')
    xlabel = 'Position error in X-O-Y (unit: m)' if is_xy else 'Yaw error (unit: degree)'
    draw_line_chart(data_list, title=f'Evaluation result on {get_dataset_name(dataset_name)} ({type_str})', xlabel=xlabel,
                    ylabel='Success Ratio(%)', save_filepath=save_filepath, yrange=[min_suc_ratio, 100], ystep=10)
    # draw traj
    data_list = []
    ## GT
    data_gt = ChartContent()
    data_gt.name = 'GT'
    data_gt.fig_color = 'black'
    for res in results:
        if res.test_name == 'scoring' or res.test_name == 'necessity':
            continue
        if res.dataset_name == dataset_name and res.loc_mode == 0 and res.use_sgv == False:
            data_gt.data = res.gt_poses
            break
    data_list.append(data_gt)
    ## PF
    data_pf = ChartContent()
    data_pf.name = 'PF-loc'
    data_pf.fig_color = 'blue'
    for res in results:
        if res.test_name == 'scoring' or res.test_name == 'necessity':
            continue
        if res.dataset_name == dataset_name and res.loc_mode == 0 and res.use_sgv == False:
            data_pf.data = res.est_poses[res.position_errs < 100]
            break
    data_list.append(data_pf)
    ## PF-SGV
    data_pf_sgv = ChartContent()
    data_pf_sgv.name = 'PF-SGV-loc'
    data_pf_sgv.fig_color = 'green'
    for res in results:
        if res.test_name == 'scoring' or res.test_name == 'necessity':
            continue
        if res.dataset_name == dataset_name and res.loc_mode == 0 and res.use_sgv == True and res.use_yaw == False:
            data_pf_sgv.data = res.est_poses[res.position_errs < 100]
            break
    data_list.append(data_pf_sgv)
    ## PF-SGV2
    data_pf_sgv2 = ChartContent()
    data_pf_sgv2.name = 'PF-SGV2-loc'
    data_pf_sgv2.fig_color = 'chocolate'
    for res in results:
        if res.test_name == 'scoring' or res.test_name == 'necessity':
            continue
        if res.dataset_name == dataset_name and res.loc_mode == 0 and res.use_sgv == True and res.use_yaw == True:
            data_pf_sgv2.data = res.est_poses[res.position_errs < 100]
            break
    data_list.append(data_pf_sgv2)
    ## Reg
    data_reg = ChartContent()
    data_reg.name = 'Reg-loc'
    data_reg.fig_color = 'cyan'
    for res in results:
        if res.test_name == 'scoring' or res.test_name == 'necessity':
            continue
        if res.dataset_name == dataset_name and res.loc_mode == -1 and res.use_sgv == True and res.use_yaw == True and np.abs(res.min_reliable_value - min_reliable_value) < 1e-10:
            data_reg.data = res.est_poses[res.position_errs < 100]
            break
    data_list.append(data_reg)
    ## Reliable
    data_reliable = ChartContent()
    data_reliable.name = 'Reliable-loc (ours)'
    data_reliable.fig_color = 'red'
    for res in results:
        if res.test_name == 'scoring' or res.test_name == 'necessity':
            continue
        if res.dataset_name == dataset_name and res.loc_mode == 1 and res.use_sgv == True and res.use_yaw == True and np.abs(res.min_reliable_value - min_reliable_value) < 1e-10:
            data_reliable.data = res.est_poses[res.position_errs < 100]
            break
    data_list.append(data_reliable)
    save_filepath = os.path.join(fig_dir, f'quantitative_eval_{dataset_name}_traj.svg')
    draw_traj(data_list, title='GT and estimated trajectories', save_filepath=save_filepath, font_size=6, transparent=True)


def draw_param_analysis(results, dataset_name, is_xy, fig_dir):
    # draw success ratio
    data_list = []
    min_reliable_values = [0.0001, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.016]
    colors = ['blue', 'green', 'red', 'cyan', 'yellow', 'chocolate', 'magenta']
    markers = ['v', 's', 'x', '^', 'd', 'o', '*']
    min_suc_ratio = 100
    for i in range(len(min_reliable_values)):
        data_reliable = ChartContent()
        data_reliable.name = f'Reliable-loc {min_reliable_values[i]:<.4f}'
        data_reliable.fig_color = colors[i]
        data_reliable.fig_marker = markers[i]
        for res in results:
            if res.test_name != 'param':
                continue
            if res.dataset_name == dataset_name and res.loc_mode == 1 and res.use_sgv == True and res.use_yaw == True and np.abs(res.min_reliable_value - min_reliable_values[i]) < 1e-10:
                if is_xy:
                    data_reliable.data = np.array(res.success_ratio_xy)
                else:
                    data_reliable.data = np.array(res.success_ratio_yaw)
                break
        min_suc_ratio = np.minimum(min_suc_ratio, data_reliable.data[0])
        data_list.append(data_reliable)
    min_suc_ratio = np.floor(min_suc_ratio / 10) * 10
    type_str = 'xy' if is_xy else 'yaw'
    save_filepath = os.path.join(fig_dir, f'param_analysis_{dataset_name}_{type_str}.svg')
    xlabel = 'Position error in X-O-Y (unit: m)' if is_xy else 'Yaw error (unit: degree)'
    draw_line_chart(data_list, title=f'Evaluation result on {get_dataset_name(dataset_name)} ({type_str})', xlabel=xlabel,
                    ylabel='Success Ratio(%)', save_filepath=save_filepath, yrange=[min_suc_ratio, 100], ystep=10)
    # draw traj
    data_list = []
    ## GT
    data_gt = ChartContent()
    data_gt.name = 'GT'
    data_gt.fig_color = 'black'
    for res in results:
        if res.test_name != 'param':
            continue
        if res.dataset_name == dataset_name and res.loc_mode == 1 and res.use_sgv == True and res.use_yaw == True and np.abs(res.min_reliable_value - min_reliable_values[0]) < 1e-10:
            data_gt.data = res.gt_poses
            break
    data_list.append(data_gt)
    ## estimattion
    for i in range(len(min_reliable_values)):
        data_reliable = ChartContent()
        data_reliable.name = f'Reliable-loc {min_reliable_values[i]:<.4f}'
        data_reliable.fig_color = colors[i]
        for res in results:
            if res.test_name != 'param':
                continue
            if res.dataset_name == dataset_name and res.loc_mode == 1 and res.use_sgv == True and res.use_yaw == True and np.abs(res.min_reliable_value - min_reliable_values[i]) < 1e-10:
                data_reliable.data = res.est_poses[res.position_errs < 100]
                break
        data_list.append(data_reliable)
    save_filepath = os.path.join(fig_dir, f'param_analysis_{dataset_name}_traj.svg')
    draw_traj(data_list, title='GT and estimated trajectories', save_filepath=save_filepath, font_size=6, transparent=True)


def draw_necessity_analysis(results, dataset_name, min_reliable_value, fig_dir):
    # success ratio
    data_list = []
    min_suc_ratio = 100
    ## reg - xy
    data_reg_xy = ChartContent()
    data_reg_xy.name = f'Reg-loc-xy'
    data_reg_xy.fig_color = 'green'
    data_reg_xy.fig_marker = 'v'
    for res in results:
        if res.test_name != 'necessity':
            continue
        if res.dataset_name == dataset_name and res.loc_mode == -1 and res.use_sgv == True and res.use_yaw == True and np.abs(res.min_reliable_value - min_reliable_value) < 1e-10:
            data_reg_xy.data = np.array(res.success_ratio_xy)
            break
    min_suc_ratio = np.minimum(min_suc_ratio, data_reg_xy.data[0])
    data_list.append(data_reg_xy)
    ## reg - yaw
    data_reg_yaw = ChartContent()
    data_reg_yaw.name = f'Reg-loc-yaw'
    data_reg_yaw.fig_color = 'green'
    data_reg_yaw.fig_marker = 's'
    for res in results:
        if res.test_name != 'necessity':
            continue
        if res.dataset_name == dataset_name and res.loc_mode == -1 and res.use_sgv == True and res.use_yaw == True and np.abs(res.min_reliable_value - min_reliable_value) < 1e-10:
            data_reg_yaw.data = np.array(res.success_ratio_yaw)
            break
    min_suc_ratio = np.minimum(min_suc_ratio, data_reg_yaw.data[0])
    data_list.append(data_reg_yaw)
    ## reliable - xy
    data_reliable_xy = ChartContent()
    data_reliable_xy.name = f'Reliable-loc-xy'
    data_reliable_xy.fig_color = 'red'
    data_reliable_xy.fig_marker = 'v'
    for res in results:
        if res.test_name != 'necessity':
            continue
        if res.dataset_name == dataset_name and res.loc_mode == 1 and res.use_sgv == True and res.use_yaw == True and np.abs(res.min_reliable_value - min_reliable_value) < 1e-10:
            data_reliable_xy.data = np.array(res.success_ratio_xy)
            break
    min_suc_ratio = np.minimum(min_suc_ratio, data_reliable_xy.data[0])
    data_list.append(data_reliable_xy)
    ## reliable - yaw
    data_reliable_yaw = ChartContent()
    data_reliable_yaw.name = f'Reliable-loc-yaw'
    data_reliable_yaw.fig_color = 'red'
    data_reliable_yaw.fig_marker = 's'
    for res in results:
        if res.test_name != 'necessity':
            continue
        if res.dataset_name == dataset_name and res.loc_mode == 1 and res.use_sgv == True and res.use_yaw == True and np.abs(res.min_reliable_value - min_reliable_value) < 1e-10:
            data_reliable_yaw.data = np.array(res.success_ratio_yaw)
            break
    min_suc_ratio = np.minimum(min_suc_ratio, data_reliable_yaw.data[0])
    data_list.append(data_reliable_yaw)
    min_suc_ratio = np.floor(min_suc_ratio / 10) * 10
    save_filepath = os.path.join(fig_dir, f'necessity_{dataset_name}.svg')
    xlabel = 'Position error in X-O-Y (unit: m) / Yaw error (unit: degree)'
    draw_line_chart(data_list, title=f'Evaluation result on {get_dataset_name(dataset_name)}', xlabel=xlabel,
                    ylabel='Success Ratio(%)', save_filepath=save_filepath, yrange=[min_suc_ratio, 100], ystep=10)
    # draw traj
    data_list = []
    ## GT
    data_gt = ChartContent()
    data_gt.name = 'GT'
    data_gt.fig_color = 'black'
    for res in results:
        if res.test_name != 'necessity':
            continue
        if res.dataset_name == dataset_name and res.loc_mode == -1 and res.use_sgv == True and res.use_yaw == True and np.abs(res.min_reliable_value - min_reliable_value) < 1e-10:
            data_gt.data = res.gt_poses
            break
    data_list.append(data_gt)
    ## Reg
    data_reg = ChartContent()
    data_reg.name = 'Reg-loc'
    data_reg.fig_color = 'green'
    for res in results:
        if res.test_name != 'necessity':
            continue
        if res.dataset_name == dataset_name and res.loc_mode == -1 and res.use_sgv == True and res.use_yaw == True and np.abs(res.min_reliable_value - min_reliable_value) < 1e-10:
            data_reg.data = res.est_poses[res.position_errs < 100]
            break
    data_list.append(data_reg)
    ## Reliable
    data_reliable = ChartContent()
    data_reliable.name = 'Reliable-loc'
    data_reliable.fig_color = 'red'
    for res in results:
        if res.test_name != 'necessity':
            continue
        if res.dataset_name == dataset_name and res.loc_mode == 1 and res.use_sgv == True and res.use_yaw == True and np.abs(res.min_reliable_value - min_reliable_value) < 1e-10:
            data_reliable.data = res.est_poses[res.position_errs < 100]
            break
    data_list.append(data_reliable)
    save_filepath = os.path.join(fig_dir, f'necessity_{dataset_name}_traj.svg')
    draw_traj(data_list, title='GT and estimated trajectories', save_filepath=save_filepath, transparent=True)


def draw_scoring_analysis(results, dataset_name, start_idx, end_idx, is_xy, fig_dir):
    # draw success ratio
    data_list = []
    min_suc_ratio = 100
    ## PF
    data_pf = ChartContent()
    data_pf.name = 'PF-loc'
    data_pf.fig_color = 'blue'
    data_pf.fig_marker = 'v'
    for res in results:
        if res.test_name != 'scoring' or res.start_idx != start_idx or res.end_idx != end_idx:
            continue
        if res.dataset_name == dataset_name and res.loc_mode == 0 and res.use_sgv == False:
            if is_xy:
                data_pf.data = np.array(res.success_ratio_xy)
            else:
                data_pf.data = np.array(res.success_ratio_yaw)
            break
    min_suc_ratio = np.minimum(min_suc_ratio, data_pf.data[0])
    data_list.append(data_pf)
    ## PF-SGV
    data_pf_sgv = ChartContent()
    data_pf_sgv.name = 'PF-SGV-loc'
    data_pf_sgv.fig_color = 'green'
    data_pf_sgv.fig_marker = 's'
    for res in results:
        if res.test_name != 'scoring' or res.start_idx != start_idx or res.end_idx != end_idx:
            continue
        if res.dataset_name == dataset_name and res.loc_mode == 0 and res.use_sgv == True and res.use_yaw == False:
            if is_xy:
                data_pf_sgv.data = np.array(res.success_ratio_xy)
            else:
                data_pf_sgv.data = np.array(res.success_ratio_yaw)
            break
    min_suc_ratio = np.minimum(min_suc_ratio, data_pf_sgv.data[0])
    data_list.append(data_pf_sgv)
    ## PF-SGV2
    data_pf_sgv2 = ChartContent()
    data_pf_sgv2.name = 'PF-SGV2-loc'
    data_pf_sgv2.fig_color = 'chocolate'
    data_pf_sgv2.fig_marker = 'x'
    for res in results:
        if res.test_name != 'scoring' or res.start_idx != start_idx or res.end_idx != end_idx:
            continue
        if res.dataset_name == dataset_name and res.loc_mode == 0 and res.use_sgv == True and res.use_yaw == True:
            if is_xy:
                data_pf_sgv2.data = np.array(res.success_ratio_xy)
            else:
                data_pf_sgv2.data = np.array(res.success_ratio_yaw)
            break
    min_suc_ratio = np.minimum(min_suc_ratio, data_pf_sgv2.data[0])
    data_list.append(data_pf_sgv2)
    min_suc_ratio = np.floor(min_suc_ratio / 10) * 10
    type_str = 'xy' if is_xy else 'yaw'
    save_filepath = os.path.join(fig_dir, f'scoring_analysis_{dataset_name}_{start_idx}_{end_idx}_{type_str}.svg')
    xlabel = 'Position error in X-O-Y (unit: m)' if is_xy else 'Yaw error (unit: degree)'
    draw_line_chart(data_list, title=f'Evaluation result on {get_dataset_name(dataset_name)} ({type_str})', xlabel=xlabel,
                    ylabel='Success Ratio(%)', save_filepath=save_filepath, yrange=[min_suc_ratio, 100], ystep=10)
    # draw traj
    data_list = []
    ## GT
    data_gt = ChartContent()
    data_gt.name = 'GT'
    data_gt.fig_color = 'black'
    for res in results:
        if res.test_name != 'scoring' or res.start_idx != start_idx or res.end_idx != end_idx:
            continue
        if res.dataset_name == dataset_name and res.loc_mode == 0 and res.use_sgv == False:
            data_gt.data = res.gt_poses
            break
    data_list.append(data_gt)
    ## PF
    data_pf = ChartContent()
    data_pf.name = 'PF-loc'
    data_pf.fig_color = 'blue'
    for res in results:
        if res.test_name != 'scoring' or res.start_idx != start_idx or res.end_idx != end_idx:
            continue
        if res.dataset_name == dataset_name and res.loc_mode == 0 and res.use_sgv == False:
            data_pf.data = res.est_poses[res.position_errs < 100]
            break
    data_list.append(data_pf)
    ## PF-SGV
    data_pf_sgv = ChartContent()
    data_pf_sgv.name = 'PF-SGV-loc'
    data_pf_sgv.fig_color = 'green'
    for res in results:
        if res.test_name != 'scoring' or res.start_idx != start_idx or res.end_idx != end_idx:
            continue
        if res.dataset_name == dataset_name and res.loc_mode == 0 and res.use_sgv == True and res.use_yaw == False:
            data_pf_sgv.data = res.est_poses[res.position_errs < 100]
            break
    data_list.append(data_pf_sgv)
    ## PF-SGV2
    data_pf_sgv2 = ChartContent()
    data_pf_sgv2.name = 'PF-SGV2-loc'
    data_pf_sgv2.fig_color = 'chocolate'
    for res in results:
        if res.test_name != 'scoring' or res.start_idx != start_idx or res.end_idx != end_idx:
            continue
        if res.dataset_name == dataset_name and res.loc_mode == 0 and res.use_sgv == True and res.use_yaw == True:
            data_pf_sgv2.data = res.est_poses[res.position_errs < 100]
            break
    data_list.append(data_pf_sgv2)
    save_filepath = os.path.join(fig_dir, f'scoring_analysis_{dataset_name}_{start_idx}_{end_idx}_traj.svg')
    draw_traj(data_list, title='GT and estimated trajectories', save_filepath=save_filepath, transparent=True)


def load_from_log_file(exp_dir, return_particles=False):
    log_file = os.path.join(exp_dir, 'train.log')
    if not os.path.isfile(log_file):
        return None
    with open(log_file) as file:
        lines = file.readlines()
        config = eval(lines[0])
        exp_res = ExpResult()
        if 'scoring' in exp_dir:  # scoring, param, necessity
            exp_res.test_name = 'scoring'
        elif 'param' in exp_dir:
            exp_res.test_name = 'param'
        elif 'necessity' in exp_dir:
            exp_res.test_name = 'necessity'
        exp_res.dataset_name = config['dataset']
        exp_res.start_idx = config['start_idx']
        exp_res.end_idx = config['end_idx']
        exp_res.loc_mode = config['loc_mode']
        exp_res.use_sgv = config['use_sgv']
        exp_res.use_yaw = config['use_yaw']
        exp_res.min_reliable_value = config['min_reliable_value']
        exp_res.num_particles = config['num_particles']
        for i in range(len(lines)):
            strs = lines[i][:-1].split(' ')
            if strs[0] == '[xy]':
                strs = strs[1:]
                exp_res.success_ratio_xy = [float(x) for x in strs]
            elif strs[0] == '[yaw]':
                strs = strs[1:]
                exp_res.success_ratio_yaw = [float(x) for x in strs]
            elif strs[0] == '[Δxy':
                    exp_res.success_ratio_2m_5deg = float(strs[8][:-2])
            elif strs[0] == 'Loc' and strs[1] == 'Mode':
                exp_res.reg_mode_ratio = int(strs[6]) / int(strs[8])
            elif strs[0] == '[pf_loc]':
                exp_res.loc_modes.append(False)
            elif strs[0] == '[reg_loc]':
                exp_res.loc_modes.append(True)
        # gt poses
        dataset_name, start_idx, end_idx, num_particles = config['dataset'], config['start_idx'], config['end_idx'], config['num_particles']
        dataset = PlaceRecognitionDataSet(name=dataset_name, for_training=False)
        dataset.dataset.get_indices_in_dataset()
        pose_file = os.path.join(dataset.dataset.data_dir(), 'helmet_submap', 'pose.csv')
        poses = loc_utils.load_poses_whu(pose_file)
        g_offset = dataset.dataset.data_cfg['global_offset']
        for i in range(len(poses)):
            poses[i, 0, 3] -= g_offset[0,0]
            poses[i, 1, 3] -= g_offset[0,1]
            poses[i, 2, 3] -= g_offset[0,2]
        end_idx = end_idx if end_idx < len(poses) else len(poses)
        exp_res.gt_locations = poses[start_idx:end_idx, :2, 3]
        exp_res.gt_headings = []
        for idx in range(len(poses)):
            exp_res.gt_headings.append(loc_utils.euler_angles_from_rotation_matrix(poses[idx][:3, :3])[2])
        exp_res.gt_headings = exp_res.gt_headings[start_idx:end_idx]
        # estimated poses
        loc_result_file = os.path.join(exp_dir, f'localization_results_{dataset_name}_{start_idx}_{end_idx}.npz')
        if os.path.exists(loc_result_file):
            loc_results = np.load(loc_result_file, allow_pickle=True)['arr_0'].tolist()
            if 'num_occupancies' in loc_results:
                exp_res.num_occupancies = loc_results['num_occupancies']
            est_poses = []
            for frame_idx in range(start_idx, end_idx):
                particles = loc_results['particles'][frame_idx]
                # collect top 25% of particles to estimate pose
                idxes = np.argsort(particles[:, -1])[::-1]
                idxes = idxes[:int(0.25 * num_particles)]
                partial_particles = particles[idxes]
                if np.sum(partial_particles[:, -1]) == 0:
                    # partial_particles[:, -1] = 1.0
                    continue
                normalized_weight = partial_particles[:, -1] / np.sum(partial_particles[:, -1])
                est_pose = partial_particles[:, :-1].T.dot(normalized_weight.T)
                est_poses.append(est_pose)
                if return_particles:
                    exp_res.particles.append(particles)
            est_poses = np.array(est_poses)
            exp_res.est_locations = est_poses[:, :2]
            # pose err
            exp_res.position_errs = np.linalg.norm(exp_res.gt_locations - exp_res.est_locations, axis=-1)
            exp_res.yaw_errs = est_poses[:, 3] - exp_res.gt_headings
            exp_res.yaw_errs = np.minimum(np.abs(exp_res.yaw_errs), 2*np.pi - np.abs(exp_res.yaw_errs)) * 180/np.pi
        return exp_res
    return None


# FIXME: it does not work well, because we do not known the convergency status.
def compute_overall_loc_performance(results, method_name):
    xy_errs, yaw_errs = [], []
    dataset_names = ['cs_college', 'info_campus', 'zhongshan_park', 'jiefang_road', 'yanjiang_road1', 'yanjiang_road2']
    if method_name == 'PF':
        for res in results:
            if res.test_name == 'scoring' or res.test_name == 'necessity':
                continue
            if res.dataset_name in dataset_names and res.loc_mode == 0 and res.use_sgv == False:
                xy_errs.append(res.position_errs)
                yaw_errs.append(res.yaw_errs)
    elif method_name == 'PF-SGV':
        for res in results:
            if res.test_name == 'scoring' or res.test_name == 'necessity':
                continue
            if res.dataset_name in dataset_names and res.loc_mode == 0 and res.use_sgv == True and res.use_yaw == False:
                xy_errs.append(res.position_errs)
                yaw_errs.append(res.yaw_errs)
    elif method_name == 'PF-SGV2':
        for res in results:
            if res.test_name == 'scoring' or res.test_name == 'necessity':
                continue
            if res.dataset_name in dataset_names and res.loc_mode == 0 and res.use_sgv == True and res.use_yaw == True:
                xy_errs.append(res.position_errs)
                yaw_errs.append(res.yaw_errs)
    elif method_name == 'Reg':
        for res in results:
            if res.test_name == 'scoring' or res.test_name == 'necessity':
                continue
            min_reliable_value = 0.0005
            if res.dataset_name == 'yanjiang_road1':
                min_reliable_value = 0.0001
            if res.dataset_name in dataset_names and res.loc_mode == -1 and res.use_sgv == True and res.use_yaw == True and np.abs(res.min_reliable_value - min_reliable_value) < 1e-10:
                xy_errs.append(res.position_errs)
                yaw_errs.append(res.yaw_errs)
    elif method_name == 'Reliable':
        for res in results:
            if res.test_name == 'scoring' or res.test_name == 'necessity':
                continue
            min_reliable_value = 0.0005
            if res.dataset_name == 'yanjiang_road1':
                min_reliable_value = 0.0001
            if res.dataset_name == dataset_name and res.loc_mode == 1 and res.use_sgv == True and res.use_yaw == True and np.abs(res.min_reliable_value - min_reliable_value) < 1e-10:
                xy_errs.append(res.position_errs)
                yaw_errs.append(res.yaw_errs)
    xy_errs = np.concatenate(xy_errs)
    yaw_errs = np.concatenate(yaw_errs)
    print(f'----------{method_name}----------')
    # success ratio
    suc_ratio = np.sum((xy_errs < 2) & (yaw_errs < 5)) / len(xy_errs) * 100
    print(f'[Δxy < 2m, Δyaw < 5deg] success ratio: {suc_ratio:<.2f}%')
    # location error
    mean_xy_err = np.mean(xy_errs)
    mean_square_xy_error = np.mean(xy_errs * xy_errs)
    rmse_xy_err = np.sqrt(mean_square_xy_error)
    print(f'[Location Error]: {mean_xy_err:<.2f} ± {rmse_xy_err:<.2f}m')
    # yaw error
    mean_yaw_err = np.mean(yaw_errs)
    mean_square_yaw_error = np.mean(yaw_errs * yaw_errs)
    rmse_yaw_err = np.sqrt(mean_square_yaw_error)
    print(f'[Yaw Error]: {mean_yaw_err:<.2f} ± {rmse_yaw_err:<.2f}m')


def vis_exp_result_seq(exp_dir):
    # load exp results
    results = []
    sub_dirs = get_sub_dirs(exp_dir)
    sub_dirs.sort()
    for sub_dir in sub_dirs:
        exp_res = load_from_log_file(os.path.join(exp_dir, sub_dir))
        if exp_res is None:
            continue
        results.append(exp_res)
    fig_dir = os.path.join(exp_dir, 'figures')
    check_makedirs(fig_dir)
    # overall localization performance
    method_names = ['PF', 'PF-SGV', 'PF-SGV2', 'Reg', 'Reliable']
    for method_name in method_names:
        compute_overall_loc_performance(results, method_name)
    # draw quantitative evaluation result
    dataset_names = ['cs_college', 'info_campus', 'zhongshan_park', 'jiefang_road', 'yanjiang_road1', 'yanjiang_road2']
    min_reliable_values = [0.0005, 0.0005, 0.0005, 0.0005, 0.0001, 0.0005]
    for i in range(len(dataset_names)):
        draw_quantitative_eval(results, dataset_names[i], min_reliable_values[i], True, fig_dir)
        draw_quantitative_eval(results, dataset_names[i], min_reliable_values[i], False, fig_dir)
    # draw param analysis
    dataset_names = ['cs_college', 'info_campus', 'zhongshan_park', 'jiefang_road', 'yanjiang_road1', 'yanjiang_road2']
    for i in range(len(dataset_names)):
        draw_param_analysis(results, dataset_names[i], True, fig_dir)
        draw_param_analysis(results, dataset_names[i], False, fig_dir)
    # draw mode switching necessity analysis
    dataset_names = ['info_campus', 'zhongshan_park', 'jiefang_road', 'yanjiang_road1']
    min_reliable_values = [0.0005, 0.0005, 0.0005, 0.0001]
    for i in range(len(dataset_names)):
        draw_necessity_analysis(results, dataset_names[i], min_reliable_values[i], fig_dir)
        draw_necessity_analysis(results, dataset_names[i], min_reliable_values[i], fig_dir)
    # draw sgv scoring analysis
    dataset_name = 'jiefang_road'
    start_idxs = [500, 1000, 1500, 2000]
    end_idxs = [600, 1100, 1600, 2100]
    for i in range(len(start_idxs)):
        draw_scoring_analysis(results, dataset_name, start_idxs[i], end_idxs[i], True, fig_dir)
        draw_scoring_analysis(results, dataset_name, start_idxs[i], end_idxs[i], False, fig_dir)


# draw curve
def draw_line_chart_bad_case(data_list, title='', xlabel='', ylabel='', xrange=[0, 500], xstep=50,
                    yrange=[0, 2048], font_size=15, save_filepath=None):
    plt.figure(figsize=(20, 10))
    plt.grid(linestyle="--")  # 设置背景网格线为虚线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['right'].set_visible(False)  # 去掉右边框
    x = np.arange(xrange[0], xrange[0]+len(data_list[0].data))
    for data_i in data_list:
        plt.plot(x, data_i.data, marker=data_i.fig_marker, color=data_i.fig_color, label=data_i.name, linewidth=1.5)

    x = np.arange(xrange[0], xrange[0] + len(data_list[0].data) + xstep, step=xstep)
    y = np.array([0, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    plt.xticks(x, fontsize=font_size, fontweight='bold')  # 默认字体大小为10
    plt.yticks(y, fontsize=font_size, fontweight='bold')
    plt.yscale('symlog')
    plt.title(label=title, pad=20, fontsize=font_size, fontweight='bold')  # 默认字体大小为12
    plt.xlabel(xlabel=xlabel, fontsize=font_size, fontweight='bold')
    plt.ylabel(ylabel=ylabel, fontsize=font_size, fontweight='bold')
    plt.xlim(xrange[0], xrange[1])  # 设置x轴的范围
    plt.ylim(yrange[0], yrange[1])

    plt.legend()  # 显示各曲线的图例
    plt.legend(loc='upper right', numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=font_size, fontweight='bold')  # 设置图例字体的大小和粗细

    if save_filepath:
        plt.savefig(save_filepath, format='svg') # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中plt.show()
        plt.close('all')
    else:
        plt.show()


def vis_bad_case(exp_dir):
    exp_res = load_from_log_file(exp_dir)
    # draw num occupancy and position error
    data_list = []
    # num occupancy
    data_occu = ChartContent()
    data_occu.name = 'Num of occupancied grids'
    data_occu.fig_color = 'blue'
    data_occu.fig_marker = '.'
    data_occu.data = exp_res.num_occupancies
    data_list.append(data_occu)
    # position error
    data_pos_err = ChartContent()
    data_pos_err.name = 'Position Error'
    data_pos_err.fig_color = 'red'
    data_pos_err.fig_marker = '.'
    data_pos_err.data = exp_res.position_errs
    data_list.append(data_pos_err)
    # fig
    save_filepath = os.path.join(exp_dir, f'bad_case_{exp_res.dataset_name}_{exp_res.start_idx}_{exp_res.end_idx}.svg')
    draw_line_chart_bad_case(data_list, title=f'Evaluation result on {get_dataset_name(exp_res.dataset_name)}',
                             xlabel='Frame ID', ylabel='Position error (unit: m) / Num of occupancies (unit: none)',
                             xrange=[exp_res.start_idx, exp_res.end_idx], save_filepath=save_filepath, font_size=26)


# temp code
def compute_PF_Reg_loc_performance(exp_dir):
    # read log file
    log_file = os.path.join(exp_dir, 'train.log')
    if not os.path.isfile(log_file):
        return None
    with open(log_file) as file:
        lines = file.readlines()
        init_frames, converge_frames, states, xy_errs, yaw_errs, time_costs = [], [], [], [], [], []
        init_frames.append(0)
        for i in range(1, len(lines)):
            strs = lines[i][:-1].split(' ')
            if strs[0] == 'Frame:':
                states.append(1)
                xy_errs.append(float(strs[4][:-1]))
                yaw_errs.append(float(strs[7][:-1]))
                time_costs.append(float(strs[-1][:-2]))
            elif strs[0] == 'Initialize':
                init_frames.append(len(xy_errs))
            elif strs[0] == '[Check':
                converge_frames.append(len(xy_errs))
        # update converge states, convered: 1, not converged: 0
        assert len(init_frames) == len(converge_frames)
        for i in range(len(init_frames)):
            for idx in range(init_frames[i], converge_frames[i]):
                states[idx] = 0
        # statistics
        states = np.array(states)
        xy_errs = np.array(xy_errs)
        yaw_errs = np.array(yaw_errs)
        time_costs = np.array(time_costs)
        # Total
        xy_errs_Total = xy_errs[states==1]
        yaw_errs_Total = yaw_errs[states==1]
        time_costs_Total = time_costs[states==1]
        suc_ratio = np.sum((xy_errs < 2) & (yaw_errs < 5)) / len(xy_errs) * 100
        xy_errs_mean = np.mean(xy_errs_Total)
        xy_errs_std = np.sqrt(np.mean(xy_errs_Total * xy_errs_Total))
        yaw_errs_mean = np.mean(yaw_errs_Total)
        yaw_errs_std = np.sqrt(np.mean(yaw_errs_Total * yaw_errs_Total))
        time_cost_mean = np.mean(time_costs_Total)
        time_cost_std = np.std(time_costs_Total)
        print(f'[Total] success ratio: {suc_ratio:<.2f}%')
        print(f'[Total] xy err: {xy_errs_mean:<.2f} ± {xy_errs_std:<.2f}m')
        print(f'[Total] yaw err: {yaw_errs_mean:<.2f} ± {yaw_errs_std:<.2f}deg')
        print(f'[Total] time cost: {time_cost_mean:<.2f} ± {time_cost_std:<.2f}ms')
        return states, xy_errs, yaw_errs, time_costs

# temp code
def compute_reliable_loc_performace(exp_dir):
    # read log file
    log_file = os.path.join(exp_dir, 'train.log')
    if not os.path.isfile(log_file):
        return None
    with open(log_file) as file:
        lines = file.readlines()
        # -1: PF(before converge); 0: PF(after converge); 1: Reg
        pf_reinit = []
        loc_modes, states, converge_frames, xy_errs, yaw_errs, time_costs = [], [], [], [], [], []
        for i in range(1, len(lines)):
            strs = lines[i][:-1].split(' ')
            if strs[0] == '[pf_loc]':
                if strs[1] == 'Re-initialize':
                    pf_reinit.append(len(loc_modes))
                    continue
                loc_modes.append('pf')
                states.append(-1)
            elif strs[0] == '[reg_loc]':
                loc_modes.append('reg')
                states.append(1)
            else:
                if strs[0] == '[Check':
                    converge_frames.append(len(loc_modes))
                continue
            xy_errs.append(float(strs[5][:-1]))
            yaw_errs.append(float(strs[8][:-1]))
            time_costs.append(float(strs[-1][:-2]))
        # update the states of converged frames
        for conv_frame in converge_frames:
            for i in range(conv_frame, len(loc_modes)):
                if loc_modes[i] == 'reg' or i in pf_reinit:
                    break
                states[i] = 0
        # statistics
        states = np.array(states)
        xy_errs = np.array(xy_errs)
        yaw_errs = np.array(yaw_errs)
        time_costs = np.array(time_costs)
        # PF: after convergency
        xy_errs_PF = xy_errs[states==0]
        yaw_errs_PF = yaw_errs[states==0]
        time_costs_PF = time_costs[states==0]
        suc_ratio = np.sum((xy_errs_PF < 2) & (yaw_errs_PF < 5)) / len(xy_errs_PF) * 100
        xy_errs_mean = np.mean(xy_errs_PF)
        xy_errs_std = np.sqrt(np.mean(xy_errs_PF * xy_errs_PF))
        yaw_errs_mean = np.mean(yaw_errs_PF)
        yaw_errs_std = np.sqrt(np.mean(yaw_errs_PF * yaw_errs_PF))
        time_cost_mean = np.mean(time_costs_PF)
        time_cost_std = np.std(time_costs_PF)
        print(f'[PF after convergency] success ratio: {suc_ratio:<.2f}%')
        print(f'[PF after convergency] xy err: {xy_errs_mean:<.2f} ± {xy_errs_std:<.2f}m')
        print(f'[PF after convergency] yaw err: {yaw_errs_mean:<.2f} ± {yaw_errs_std:<.2f}deg')
        print(f'[PF after convergency] time cost: {time_cost_mean:<.2f} ± {time_cost_std:<.2f}ms')
        # Reg
        xy_errs_Reg = xy_errs[states==1]
        yaw_errs_Reg = yaw_errs[states==1]
        time_costs_Reg = time_costs[states==1]
        suc_ratio = np.sum((xy_errs_Reg < 2) & (yaw_errs_Reg < 5)) / len(xy_errs_Reg) * 100
        xy_errs_mean = np.mean(xy_errs_Reg)
        xy_errs_std = np.sqrt(np.mean(xy_errs_Reg * xy_errs_Reg))
        yaw_errs_mean = np.mean(yaw_errs_Reg)
        yaw_errs_std = np.sqrt(np.mean(yaw_errs_Reg * yaw_errs_Reg))
        time_cost_mean = np.mean(time_costs_Reg)
        time_cost_std = np.std(time_costs_Reg)
        print(f'[Reg] success ratio: {suc_ratio:<.2f}%')
        print(f'[Reg] xy err: {xy_errs_mean:<.2f} ± {xy_errs_std:<.2f}m')
        print(f'[Reg] yaw err: {yaw_errs_mean:<.2f} ± {yaw_errs_std:<.2f}deg')
        print(f'[Reg] time cost: {time_cost_mean:<.2f} ± {time_cost_std:<.2f}ms')
        # Total
        xy_errs_Total = xy_errs[states!=-1]
        yaw_errs_Total = yaw_errs[states!=-1]
        time_costs_Total = time_costs[states!=-1]
        suc_ratio = np.sum((xy_errs < 2) & (yaw_errs < 5)) / len(xy_errs) * 100
        xy_errs_mean = np.mean(xy_errs_Total)
        xy_errs_std = np.sqrt(np.mean(xy_errs_Total * xy_errs_Total))
        yaw_errs_mean = np.mean(yaw_errs_Total)
        yaw_errs_std = np.sqrt(np.mean(yaw_errs_Total * yaw_errs_Total))
        time_cost_mean = np.mean(time_costs_Total)
        time_cost_std = np.std(time_costs_Total)
        print(f'[Total] success ratio: {suc_ratio:<.2f}%')
        print(f'[Total] xy err: {xy_errs_mean:<.2f} ± {xy_errs_std:<.2f}m')
        print(f'[Total] yaw err: {yaw_errs_mean:<.2f} ± {yaw_errs_std:<.2f}deg')
        print(f'[Total] time cost: {time_cost_mean:<.2f} ± {time_cost_std:<.2f}ms')
        return states, xy_errs, yaw_errs, time_costs

# temp code
def compute_PF_Reg_Reliable_performance():
    # PF
    print('----------PF----------')
    exp_dirs = [
        '/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/result3/2024-03-22T21-44-05_cs_college_only_pf',
        '/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/result3/2024-03-22T22-01-56_info_campus_only_pf',
        '/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/result3/2024-03-22T23-20-42_zhongshan_park_only_pf',
        '/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/result3/2024-03-23T00-15-19_jiefang_road_only_pf',
        '/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/result3/2024-03-23T01-00-08_yanjiang_road1_only_pf',
        '/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/result3/2024-03-23T01-28-24_yanjiang_road2_only_pf'
    ]
    states, xy_errs, yaw_errs, time_costs = [], [], [], []
    for exp_dir in exp_dirs:
        states_i, xy_errs_i, yaw_errs_i, time_costs_i = compute_PF_Reg_loc_performance(exp_dir)
        states.append(states_i)
        xy_errs.append(xy_errs_i)
        yaw_errs.append(yaw_errs_i)
        time_costs.append(time_costs_i)
    states = np.concatenate(states)
    xy_errs = np.concatenate(xy_errs)
    yaw_errs = np.concatenate(yaw_errs)
    time_costs = np.concatenate(time_costs)
    xy_errs_Total = xy_errs[states==1]
    yaw_errs_Total = yaw_errs[states==1]
    time_costs_Total = time_costs[states==1]
    suc_ratio = np.sum((xy_errs < 2) & (yaw_errs < 5)) / len(xy_errs) * 100
    xy_errs_mean = np.mean(xy_errs_Total)
    xy_errs_std = np.sqrt(np.mean(xy_errs_Total * xy_errs_Total))
    yaw_errs_mean = np.mean(yaw_errs_Total)
    yaw_errs_std = np.sqrt(np.mean(yaw_errs_Total * yaw_errs_Total))
    time_cost_mean = np.mean(time_costs_Total)
    time_cost_std = np.std(time_costs_Total)
    print(f'[Mean] success ratio: {suc_ratio:<.2f}%')
    print(f'[Mean] xy err: {xy_errs_mean:<.2f} ± {xy_errs_std:<.2f}m')
    print(f'[Mean] yaw err: {yaw_errs_mean:<.2f} ± {yaw_errs_std:<.2f}deg')
    print(f'[Mean] time cost: {time_cost_mean:<.2f} ± {time_cost_std:<.2f}ms')
    # PF-SGV
    print('----------PF-SGV----------')
    exp_dirs = [
        '/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/result3/2024-03-22T21-49-04_cs_college_only_pf_sgv',
        '/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/result3/2024-03-22T22-28-50_info_campus_only_pf_sgv',
        '/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/result3/2024-03-22T23-36-26_zhongshan_park_only_pf_sgv',
        '/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/result3/2024-03-23T00-29-23_jiefang_road_only_pf_sgv',
        '/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/result3/2024-03-23T01-07-59_yanjiang_road1_only_pf_sgv',
        '/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/result3/2024-03-23T01-40-14_yanjiang_road2_only_pf_sgv'
    ]
    states, xy_errs, yaw_errs, time_costs = [], [], [], []
    for exp_dir in exp_dirs:
        states_i, xy_errs_i, yaw_errs_i, time_costs_i = compute_PF_Reg_loc_performance(exp_dir)
        states.append(states_i)
        xy_errs.append(xy_errs_i)
        yaw_errs.append(yaw_errs_i)
        time_costs.append(time_costs_i)
    states = np.concatenate(states)
    xy_errs = np.concatenate(xy_errs)
    yaw_errs = np.concatenate(yaw_errs)
    time_costs = np.concatenate(time_costs)
    xy_errs_Total = xy_errs[states==1]
    yaw_errs_Total = yaw_errs[states==1]
    time_costs_Total = time_costs[states==1]
    suc_ratio = np.sum((xy_errs < 2) & (yaw_errs < 5)) / len(xy_errs) * 100
    xy_errs_mean = np.mean(xy_errs_Total)
    xy_errs_std = np.sqrt(np.mean(xy_errs_Total * xy_errs_Total))
    yaw_errs_mean = np.mean(yaw_errs_Total)
    yaw_errs_std = np.sqrt(np.mean(yaw_errs_Total * yaw_errs_Total))
    time_cost_mean = np.mean(time_costs_Total)
    time_cost_std = np.std(time_costs_Total)
    print(f'[Mean] success ratio: {suc_ratio:<.2f}%')
    print(f'[Mean] xy err: {xy_errs_mean:<.2f} ± {xy_errs_std:<.2f}m')
    print(f'[Mean] yaw err: {yaw_errs_mean:<.2f} ± {yaw_errs_std:<.2f}deg')
    print(f'[Mean] time cost: {time_cost_mean:<.2f} ± {time_cost_std:<.2f}ms')
    # PF-SGV2
    print('----------PF-SGV2----------')
    exp_dirs = [
        '/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/result3/2024-03-22T21-54-23_cs_college_only_pf_sgv_yaw',
        '/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/result3/2024-03-22T22-56-54_info_campus_only_pf_sgv_yaw',
        '/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/result3/2024-03-22T23-54-20_zhongshan_park_only_pf_sgv_yaw',
        '/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/result3/2024-03-23T00-43-24_jiefang_road_only_pf_sgv_yaw',
        '/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/result3/2024-03-23T01-17-08_yanjiang_road1_only_pf_sgv_yaw',
        '/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/result3/2024-03-23T01-55-03_yanjiang_road2_only_pf_sgv_yaw'
    ]
    states, xy_errs, yaw_errs, time_costs = [], [], [], []
    for exp_dir in exp_dirs:
        states_i, xy_errs_i, yaw_errs_i, time_costs_i = compute_PF_Reg_loc_performance(exp_dir)
        states.append(states_i)
        xy_errs.append(xy_errs_i)
        yaw_errs.append(yaw_errs_i)
        time_costs.append(time_costs_i)
    states = np.concatenate(states)
    xy_errs = np.concatenate(xy_errs)
    yaw_errs = np.concatenate(yaw_errs)
    time_costs = np.concatenate(time_costs)
    xy_errs_Total = xy_errs[states==1]
    yaw_errs_Total = yaw_errs[states==1]
    time_costs_Total = time_costs[states==1]
    suc_ratio = np.sum((xy_errs < 2) & (yaw_errs < 5)) / len(xy_errs) * 100
    xy_errs_mean = np.mean(xy_errs_Total)
    xy_errs_std = np.sqrt(np.mean(xy_errs_Total * xy_errs_Total))
    yaw_errs_mean = np.mean(yaw_errs_Total)
    yaw_errs_std = np.sqrt(np.mean(yaw_errs_Total * yaw_errs_Total))
    time_cost_mean = np.mean(time_costs_Total)
    time_cost_std = np.std(time_costs_Total)
    print(f'[Mean] success ratio: {suc_ratio:<.2f}%')
    print(f'[Mean] xy err: {xy_errs_mean:<.2f} ± {xy_errs_std:<.2f}m')
    print(f'[Mean] yaw err: {yaw_errs_mean:<.2f} ± {yaw_errs_std:<.2f}deg')
    print(f'[Mean] time cost: {time_cost_mean:<.2f} ± {time_cost_std:<.2f}ms')
    # Reg
    print('----------Reg----------')
    exp_dirs = [
        '/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/result3/2024-03-22T21-59-34_cs_college_only_reg_0.0005',
        '/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/result3/2024-03-22T23-17-53_info_campus_only_reg_0.0005',
        '/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/result3/2024-03-23T00-12-48_zhongshan_park_only_reg_0.0005',
        '/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/result3/2024-03-23T00-57-39_jiefang_road_only_reg_0.0005',
        '/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/result3/2024-03-23T01-25-46_yanjiang_road1_only_reg_0.0001',
        '/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/result3/2024-03-23T02-08-15_yanjiang_road2_only_reg_0.0005'
    ]
    states, xy_errs, yaw_errs, time_costs = [], [], [], []
    for exp_dir in exp_dirs:
        states_i, xy_errs_i, yaw_errs_i, time_costs_i = compute_PF_Reg_loc_performance(exp_dir)
        states.append(states_i)
        xy_errs.append(xy_errs_i)
        yaw_errs.append(yaw_errs_i)
        time_costs.append(time_costs_i)
    states = np.concatenate(states)
    xy_errs = np.concatenate(xy_errs)
    yaw_errs = np.concatenate(yaw_errs)
    time_costs = np.concatenate(time_costs)
    xy_errs_Total = xy_errs[states==1]
    yaw_errs_Total = yaw_errs[states==1]
    time_costs_Total = time_costs[states==1]
    suc_ratio = np.sum((xy_errs < 2) & (yaw_errs < 5)) / len(xy_errs) * 100
    xy_errs_mean = np.mean(xy_errs_Total)
    xy_errs_std = np.sqrt(np.mean(xy_errs_Total * xy_errs_Total))
    yaw_errs_mean = np.mean(yaw_errs_Total)
    yaw_errs_std = np.sqrt(np.mean(yaw_errs_Total * yaw_errs_Total))
    time_cost_mean = np.mean(time_costs_Total)
    time_cost_std = np.std(time_costs_Total)
    print(f'[Mean] success ratio: {suc_ratio:<.2f}%')
    print(f'[Mean] xy err: {xy_errs_mean:<.2f} ± {xy_errs_std:<.2f}m')
    print(f'[Mean] yaw err: {yaw_errs_mean:<.2f} ± {yaw_errs_std:<.2f}deg')
    print(f'[Mean] time cost: {time_cost_mean:<.2f} ± {time_cost_std:<.2f}ms')
    # Reliable
    print('----------Reliable----------')
    exp_dirs = [
        '/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/result3/2024-03-22T16-22-58_cs_college_reliable_0.0005_param',
        '/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/result3/2024-03-22T16-48-18_info_campus_reliable_0.0005_param',
        '/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/result3/2024-03-22T17-30-55_zhongshan_park_reliable_0.0005_param',
        '/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/result3/2024-03-22T18-23-40_jiefang_road_reliable_0.0005_param',
        '/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/result3/2024-03-22T19-21-42_yanjiang_road1_reliable_0.0001_param',
        '/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/result3/2024-03-22T20-24-43_yanjiang_road2_reliable_0.0005_param',
    ]
    states, xy_errs, yaw_errs, time_costs = [], [], [], []
    for exp_dir in exp_dirs:
        states_i, xy_errs_i, yaw_errs_i, time_costs_i = compute_reliable_loc_performace(exp_dir)
        states.append(states_i)
        xy_errs.append(xy_errs_i)
        yaw_errs.append(yaw_errs_i)
        time_costs.append(time_costs_i)
    states = np.concatenate(states)
    xy_errs = np.concatenate(xy_errs)
    yaw_errs = np.concatenate(yaw_errs)
    time_costs = np.concatenate(time_costs)
    xy_errs_Total = xy_errs[states!=-1]
    yaw_errs_Total = yaw_errs[states!=-1]
    time_costs_Total = time_costs[states!=-1]
    suc_ratio = np.sum((xy_errs < 2) & (yaw_errs < 5)) / len(xy_errs) * 100
    xy_errs_mean = np.mean(xy_errs_Total)
    xy_errs_std = np.sqrt(np.mean(xy_errs_Total * xy_errs_Total))
    yaw_errs_mean = np.mean(yaw_errs_Total)
    yaw_errs_std = np.sqrt(np.mean(yaw_errs_Total * yaw_errs_Total))
    time_cost_mean = np.mean(time_costs_Total)
    time_cost_std = np.std(time_costs_Total)
    print(f'[Mean] success ratio: {suc_ratio:<.2f}%')
    print(f'[Mean] xy err: {xy_errs_mean:<.2f} ± {xy_errs_std:<.2f}m')
    print(f'[Mean] yaw err: {yaw_errs_mean:<.2f} ± {yaw_errs_std:<.2f}deg')
    print(f'[Mean] time cost: {time_cost_mean:<.2f} ± {time_cost_std:<.2f}ms')


if __name__ == '__main__':
    # temp code
    compute_PF_Reg_Reliable_performance()
    
    # # vis bad case
    # exp_dir = '/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/2024-03-24T17-58-36_jiefang_road_reliable_0.0005'
    # vis_bad_case(exp_dir)
    
    # # vis exp res
    # exp_dir = '/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/result3'
    # vis_exp_result_seq(exp_dir)
    
    # logger
    exp_dir = '/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/2024-04-23T20-28-27_info_campus_reliable_0.0005'
    log_dir = os.path.join(exp_dir, 'vis_log')
    check_makedirs(log_dir)
    logger = loc_utils.get_logger(log_dir)
    
    # params
    config = get_args()
    with open(os.path.join(exp_dir, 'train.log')) as file:
        line = file.readline().replace("'", '"')
        file.close()
        config = eval(line[:-1])

    # setup parameters
    numParticles = config['num_particles']
    save_result = config['save_result']
    move_thres = config['move_thres']

    # load dataset
    dataset = PlaceRecognitionDataSet(name=config['dataset'], for_training=False)
    dataset.dataset.get_indices_in_dataset()

    # load helmet info and poses
    query_trip_idx = dataset.dataset.get_trip_idx_by_name("helmet_submap")
    query_indices = dataset.dataset.sample_indices[query_trip_idx]
    pose_file = os.path.join(dataset.dataset.data_dir(), 'helmet_submap', 'pose.csv')
    poses = loc_utils.load_poses_whu(pose_file)
    g_offset = dataset.dataset.data_cfg['global_offset']
    for i in range(len(poses)):
        poses[i, 0, 3] -= g_offset[0,0]
        poses[i, 1, 3] -= g_offset[0,1]
        poses[i, 2, 3] -= g_offset[0,2]
    
    # initialize sensor model
    sensor_model = SensorModel(logger, config, dataset.dataset, poses)
    sensor_model.construct_map_tree()
    grid_coords = sensor_model.map_positions
    map_size = [np.min(grid_coords[:, 0]), np.max(grid_coords[:, 0]),
                np.min(grid_coords[:, 1]), np.max(grid_coords[:, 1])]
    
    # generate motion commands
    commands, gt_headings = MotionModel.gen_commands(poses)

    # load results
    start_idx = config['start_idx']
    end_idx = config['end_idx']
    end_idx = end_idx if end_idx < len(poses) else len(poses)
    dataset_name = config['dataset']
    result_file = os.path.join(exp_dir, f'localization_results_{dataset_name}_{start_idx}_{end_idx}.npz')
    if os.path.exists(result_file):
        results = np.load(result_file, allow_pickle=True)['arr_0'].tolist()
    else:
        print('result file does not exists at: ', result_file)
        exit(-1)

    # test trajectory plotting
    plot_traj_result(results, poses, gt_headings, numParticles=numParticles, start_idx=start_idx, end_idx=end_idx, logger=logger)

    # test offline visualizer
    vis_offline(results, poses, gt_headings, map_size, numParticles=numParticles, start_idx=start_idx)
