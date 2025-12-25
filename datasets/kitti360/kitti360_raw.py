# Functions and classes operating on a raw Kitti datasets
# This script is adapted from: https://github.com/jac99/Egonn/blob/main/datasets/kitti/kitti_raw.py

import numpy as np
import os
import sys
import torch
from torch.utils.data import Dataset
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from datasets.point_clouds_utils import PointCloudLoader


class Kitti360PointCloudLoader(PointCloudLoader):
    def set_properties(self):
        # Set point cloud properties, such as ground_plane_level.
        self.ground_plane_level = -1.5

    def read_pc(self, file_pathname: str) -> torch.Tensor:
        # Reads the point cloud without pre-processing
        # Returns Nx3 tensor
        pc = np.fromfile(file_pathname, dtype=np.float32)
        # PC in Mulran is of size [num_points, 4] -> x,y,z,reflectance
        pc = np.reshape(pc, (-1, 4))[:, :3]
        return pc


class Kitti360Sequence(Dataset):
    """
    Point cloud from a sequence from a raw Mulran datasets
    """
    def __init__(self, dataset_root: str, sequence_name: str, pose_time_tolerance: float = 1.,
                 remove_zero_points: bool = True):
        # pose_time_tolerance: (in seconds) skip point clouds without corresponding pose information (based on
        #                      timestamps difference)
        # remove_zero_points: remove (0,0,0) points

        assert os.path.exists(dataset_root), f'Cannot access datasets root: {dataset_root}'
        self.dataset_root = dataset_root
        self.sequence_name = '2013_05_28_drive_00'+ sequence_name + '_sync'
        # self.sequence_path = os.path.join(self.dataset_root, 'sequences')
        # assert os.path.exists(self.sequence_path), f'Cannot access sequence: {self.sequence_path}'
        self.rel_lidar_path = os.path.join(self.sequence_name, 'velodyne_points/data')
        # lidar_path = os.path.join(self.sequence_path, self.rel_lidar_path)
        # assert os.path.exists(lidar_path), f'Cannot access lidar scans: {lidar_path}'
        self.pose_file = os.path.join(self.dataset_root, self.sequence_name , 'poses.txt')
        self.calib_file = os.path.join(self.dataset_root, self.sequence_name , 'cam0_to_world.txt')
        assert os.path.exists(self.pose_file), f'Cannot access sequence pose file: {self.pose_file}'
        self.times_file = os.path.join(self.dataset_root, self.sequence_name, 'velodyne_points/timestamps.txt')
        assert os.path.exists(self.pose_file), f'Cannot access sequence times file: {self.times_file}'
        # Maximum discrepancy between timestamps of LiDAR scan and global pose in seconds
        self.pose_time_tolerance = pose_time_tolerance
        self.remove_zero_points = remove_zero_points

        self.rel_lidar_timestamps, self.lidar_poses, filenames = self._read_lidar_poses()
        self.rel_scan_filepath = [os.path.join(self.rel_lidar_path, '%010d%s' % (e, '.bin')) for e in filenames]
        print('')

    def __len__(self):
        return len(self.rel_lidar_timestamps)

    def __getitem__(self, ndx):
        scan_filepath = os.path.join(self.dataset_root, self.rel_scan_filepath[ndx])
        pc = load_pc(scan_filepath)
        if self.remove_zero_points:
            mask = np.all(np.isclose(pc, 0), axis=1)
            pc = pc[~mask]
        return {'pc': pc, 'pose': self.lidar_poses[ndx], 'ts': self.rel_lidar_timestamps[ndx]}

    def _read_lidar_poses(self):
        fnames = os.listdir(os.path.join(self.dataset_root, self.rel_lidar_path))
        temp = os.path.join(self.dataset_root, self.rel_lidar_path)
        fnames = [e for e in fnames if os.path.isfile(os.path.join(temp, e))]
        assert len(fnames) > 0, f"Make sure that the path {self.rel_lidar_path}"
        # filenames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])
        # with open(self.calib_file, 'r') as f:
        #     for line in f.readlines():
        #         data = np.array([float(x) for x in line.split()])

        # cam0_to_velo = np.reshape(data, (3, 4))
        # cam0_to_velo = np.vstack([cam0_to_velo, [0, 0, 0, 1]])


        poses,_,_ = load_poses_from_txt(self.pose_file)#, cam0_to_velo)
        sorted_keys = sorted(poses.keys())
        poses_list =  [poses[k] for k in sorted_keys]
        filenames = sorted([int(key) for key in poses])
        ts = load_timestamps(self.times_file)
        ts = np.asarray(ts)[filenames]
        rel_ts = ts - ts[0]

        return rel_ts, poses_list, filenames


def load_pc(filepath):
    # Load point cloud, does not apply any transform
    # Returns Nx3 matrix
    pc = np.fromfile(filepath, dtype=np.float32)
    # PC in Kitti is of size [num_points, 4] -> x,y,z,reflectance
    pc = np.reshape(pc, (-1, 4))[:, :3]
    return pc

def load_poses_from_txt(file_name):#, cam0_to_velo):
    f = open(file_name, 'r')
    s = f.readlines()
    f.close()
    transforms = {}
    x = []
    y = []
    for cnt, line in enumerate(s):
        P = np.eye(4)
        line_split = [float(i) for i in line.split(" ") if i!="" and i!="\n"]
        withIdx = len(line_split) >= 13
        for row in range(3):
            for col in range(4):
                P[row, col] = line_split[row*4 + col + withIdx]
        if withIdx:
            frame_idx = line_split[0]
        else:
            frame_idx = cnt
        transforms[frame_idx] = P #@ cam0_to_velo.inverse()
        x.append(P[0, 3])
        y.append(P[1, 3])
    return transforms, x, y

def load_timestamps(file_name):
    f = open(file_name, 'r')
    s = f.readlines()
    f.close()
    times = []

    for cnt, line in enumerate(s):#2013-05-28 11:36:55.89086054
        dt_obj = datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')

        # nanosec = dt_obj.timestamp() * 10e9
        sec = dt_obj.timestamp() 

        times.append(sec)
    return times