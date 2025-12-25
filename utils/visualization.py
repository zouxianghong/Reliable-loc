import copy
import os

import open3d as o3d
import numpy as np

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

from utils.loading_pointclouds import normalize_point_cloud, load_pc_file


def get_color_map(x):
    colours = plt.cm.Spectral(x)
    return colours[:, :3]


def mesh_sphere(pcd, voxel_size, sphere_size=0.6):
    # Create a mesh sphere
    spheres = o3d.geometry.TriangleMesh()
    s = o3d.geometry.TriangleMesh.create_sphere(radius=voxel_size * sphere_size)
    s.compute_vertex_normals()

    for i, p in enumerate(pcd.points):
        si = copy.deepcopy(s)
        trans = np.identity(4)
        trans[:3, 3] = p
        si.transform(trans)
        si.paint_uniform_color(pcd.colors[i])
        spheres += si
    return spheres


def get_colored_point_cloud_feature(pcd, feature, voxel_size):
    tsne_results = embed_tsne(feature)
    # color = get_color_map(tsne_results)
    color = tsne_results
    pcd.colors = o3d.utility.Vector3dVector(color)
    spheres = mesh_sphere(pcd, voxel_size)
    return spheres


def embed_tsne(data):
    """
    N x D np.array data
    """
    tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300, random_state=0, n_jobs=2)
    tsne_results = tsne.fit_transform(data)
    tsne_results = np.squeeze(tsne_results)
    tsne_min = np.min(tsne_results)
    tsne_max = np.max(tsne_results)
    return (tsne_results - tsne_min) / (tsne_max - tsne_min)


def visualize_feature_embedding(coor, feature):
    vis_pcd = o3d.geometry.PointCloud()
    vis_pcd.points = o3d.utility.Vector3dVector(coor)
    vis_pcd = get_colored_point_cloud_feature(vis_pcd, feature, 0.02)
    o3d.visualization.draw_geometries([vis_pcd])


def visualize_multi_feature_embedding(coor_feature_list):
    geometries = []
    for coor, feature in coor_feature_list:
        vis_pcd = o3d.geometry.PointCloud()
        vis_pcd.points = o3d.utility.Vector3dVector(coor)
        vis_pcd = get_colored_point_cloud_feature(vis_pcd, feature, 0.02)
        geometries.append(vis_pcd)
    o3d.visualization.draw_geometries(geometries)


def visualize_point_cloud(wnd_name, coor, colors):
    vis = o3d.visualization.Visualizer()
    # 设置窗口标题
    vis.create_window(window_name=wnd_name)
    # 设置点云大小
    vis.get_render_option().point_size = 3
    # 设置颜色背景为黑色
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    # 创建点云对象
    pcd = o3d.open3d.geometry.PointCloud()
    # 将点云数据转换为Open3d可以直接使用的数据类型
    pcd.points = o3d.open3d.utility.Vector3dVector(coor)
    pcd.colors = o3d.open3d.utility.Vector3dVector(colors)
    # 将点云加入到窗口中
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()


def vis_cloud_simple(wnd_name, coor_list):
    colors = [np.array([1.0, 1.0, 1.0]).reshape([1, 3]), np.array([1.0, 0.0, 0.0]).reshape([1, 3]),
              np.array([0.0, 1.0, 0.0]).reshape([1, 3]), np.array([0.0, 0.0, 1.0]).reshape([1, 3]),
              np.array([1.0, 1.0, 0.0]).reshape([1, 3]), np.array([0.0, 1.0, 1.0]).reshape([1, 3])]
    pcs, feats = [], []
    for i in range(len(coor_list)):
        coor = np.array(coor_list[i], dtype=np.float32)
        feat = np.repeat(colors[i], coor.shape[0], axis=0)
        pcs.append(coor)
        feats.append(feat)
    vis_pc = np.concatenate(pcs, axis=0)
    vis_feat = np.concatenate(feats, axis=0)
    visualize_point_cloud(wnd_name, vis_pc, vis_feat)


class BadCase:
    def __init__(self):
        self.query_file = ''
        self.pos_files = []
        self.neg_files = []


def load_bad_case(bad_case_file):
    bad_cases_top1p, bad_case_top1 = [], []
    load_top1 = False
    temp_bad_case = None
    for line in open(bad_case_file, 'r'):
        line = line.strip()
        if line == '--------------------BadCases: top1--------------------':
            load_top1 = True
        elif 'query' in line:
            temp_bad_case = BadCase()
            temp_bad_case.query_file = line.strip(' ')[2]
        elif 'true' in line:
            lines = line.strip(' ')
            for i in range(2, len(lines)):
                temp_bad_case.pos_files.append(lines[i])
        elif 'false' in line:
            lines = line.strip(' ')
            for i in range(2, len(lines)):
                temp_bad_case.neg_files.append(lines[i])
            if load_top1:
                bad_case_top1.append(temp_bad_case)
            else:
                bad_cases_top1p.append(temp_bad_case)
    return bad_cases_top1p, bad_case_top1


def vis_bad_cases(bad_case_file):
    # load bad cases
    bad_cases_top1p, bad_case_top1 = load_bad_case(bad_case_file)
    # get case dir
    colors = [np.array([0.0, 0.0, 1.0]).reshape([1, 3]),  # blue
              np.array([0.0, 1.0, 0.0]).reshape([1, 3]),  # gree
              np.array([1.0, 0.0, 0.0]).reshape([1, 3])]  # red
    bad_cases_list = [bad_cases_top1p, bad_case_top1]
    wnd_name_list = ['badcase_top1%_', 'badcase_top1_']
    for bad_cases_idx in bad_cases_list:
        bad_cases = bad_cases_list[bad_cases_idx]
        for bad_case_idx in bad_cases:
            bad_case = bad_cases[bad_case_idx]
            files_list = [[bad_case.query_file], bad_case.pos_files, bad_case.neg_files]
            pcs, feats = [], []
            for i in range(len(files_list)):
                files = files_list[i]
                for f_idx in files:
                    pc = load_pc_file(files[f_idx], use_np_load=True) + np.array([f_idx, 0.0, 0.0]).reshape([1, 3])
                    coor = np.array(pc, dtype=np.float32)
                    feat = np.repeat(colors[i], coor.shape[0], axis=0)
                    pcs.append(coor)
                    feats.append(feat)
            vis_pc = np.concatenate(pcs, axis=0)
            vis_feat = np.concatenate(feats, axis=0)
            wnd_name = wnd_name_list[bad_cases_idx] + '{}'.format(bad_case_idx)
            visualize_point_cloud(wnd_name, vis_pc, vis_feat)


if __name__ == '__main__':
    # coor_feat_tuples = []
    # query_files = ["/home/ericxhzou/Code/benchmark_datasets/info_campus_all/helmet_submap/pointcloud_25m_70.4deg_30deg/4402874639.bin"]
    # for query_file in query_files:
    #     query_pc = np.load(query_file)
    #     query_pc = query_pc.reshape([-1, 3])
    #     query_pc = normalize_point_cloud(query_pc)
    #     feat = np.random.random(size=query_pc.shape)
    #     coor_feat_tuple = tuple([query_pc, feat])
    #     coor_feat_tuples.append(coor_feat_tuple)
    # visualize_multi_feature_embedding(coor_feat_tuples)

    # query_pc = np.load("/home/ericxhzou/Code/benchmark_datasets/info_campus_all/helmet_submap/pointcloud_25m_70.4deg_30deg/4402874639.bin")
    # query_pc = query_pc.reshape([-1, 3])
    # query_feat = np.random.random(size=query_pc.shape)
    # top1_ref_pc = np.load("/home/ericxhzou/Code/benchmark_datasets/info_campus_all/map_submap_along_traj/pointcloud_25m_70.4deg_30deg/24_-23_0_3.bin")
    # top1_ref_pc = top1_ref_pc.reshape([-1, 3])
    # top1_ref_feat = np.random.random(size=query_pc.shape) + 1.0
    # vis_pc = np.concatenate((query_pc, top1_ref_pc), axis=0)
    # vis_feat = np.concatenate((query_feat, top1_ref_feat), axis=0)
    # # visualize_feature_embedding(vis_pc, vis_feat)
    # visualize_point_cloud("query_top1ref", vis_pc, colors=vis_feat)

    vis_bad_cases('/home/ericxhzou/Code/test_res_cmp/1->2baseline-hankou/new_badcase.txt')
