# Test set for Kitti360 Sequence 09.
# This script is adapted from: https://github.com/jac99/Egonn/blob/main/datasets/kitti/generate_evaluation_sets.py

import numpy as np
import argparse
from typing import List
import os
import sys
import csv
import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from datasets.kitti360.kitti360_raw import Kitti360Sequence, Kitti360PointCloudLoader
from datasets.base_datasets import EvaluationTuple, EvaluationSet, filter_query_elements


# MAP_TIMERANGE = (0, 170)
MAP_TIMERANGE = (0, 300)

def get_scans(sequence: Kitti360Sequence, min_displacement: float = 0.1, ts_range: tuple = None) -> List[EvaluationTuple]:
    # Get a list of all point clouds from the sequence (the full sequence or test split only)

    elems = []
    old_pos = None
    count_skipped = 0
    displacements = []

    for ndx in range(len(sequence)):
        if ts_range is not None:
            if (ts_range[0] > sequence.rel_lidar_timestamps[ndx]) or (ts_range[1] < sequence.rel_lidar_timestamps[ndx]):
                continue
        pose = sequence.lidar_poses[ndx]
        # Kitti poses are in camera coordinates system where where y is upper axis dim
        position = pose[[0,1], 3]

        if old_pos is not None:
            displacements.append(np.linalg.norm(old_pos - position))

            if np.linalg.norm(old_pos - position) < min_displacement:
                # Ignore the point cloud if the vehicle didn't move
                count_skipped += 1
                continue

        item = EvaluationTuple(sequence.rel_lidar_timestamps[ndx], sequence.rel_scan_filepath[ndx], position, pose)
        elems.append(item)
        old_pos = position

    print(f'{count_skipped} clouds skipped due to displacement smaller than {min_displacement}')
    print(f'mean displacement {np.mean(np.array(displacements))}')
    return elems


def generate_evaluation_set(sequence: Kitti360Sequence, min_displacement: float = 0.1,
                            dist_threshold: float = 5.) -> EvaluationSet:
    map_set = get_scans(sequence, min_displacement, MAP_TIMERANGE)
    query_set = get_scans(sequence, min_displacement, (MAP_TIMERANGE[-1], sequence.rel_lidar_timestamps[-1]))
    query_set = filter_query_elements(query_set, map_set, dist_threshold)
    print(f'{len(map_set)} database elements, {len(query_set)} query elements')
    return EvaluationSet(query_set, map_set)


# create map or query sequence data, refer to Oxford RobotCar / 3-Inhouse Dataset in PointNetVLAD
def create_seq_data(dataset_root: str, seq_name: str, scan_type: str, mq_set: List[EvaluationTuple]):
    print('Create Map/Query Sequence Data just like Oxford RobotCar Dataset in PointNetVLAD')
    pc_loader = Kitti360PointCloudLoader()
    seq_dir = os.path.join(dataset_root, seq_name)
    scan_dir = os.path.join(seq_dir, scan_type)
    if not os.path.exists(scan_dir):
        os.makedirs(scan_dir)
    csv_file = scan_dir + '.csv'
    with open(csv_file, 'w', newline='') as newfile:
        writer = csv.writer(newfile)
        writer.writerow(['timestamp', 'northing', 'easting'])
        for i in tqdm.tqdm(range(len(mq_set))):
            anchor_pos = mq_set[i].position
            pc_file = os.path.join(dataset_root, mq_set[i].rel_scan_filepath)
            anchor_pc = pc_loader(pc_file)
            timestamp, _ = os.path.splitext(os.path.basename(mq_set[i].rel_scan_filepath))
            # write to csv: ts, y, x
            writer.writerow([int(timestamp), anchor_pos[1], anchor_pos[0]])
            # save point cloud to *.bin file
            pc_file = os.path.join(scan_dir, str(int(timestamp)) + '.bin')
            anchor_pc.tofile(pc_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate evaluation sets for KItti datasets')
    # kitti: /mnt/088A6CBB8A6CA742/Datasets/Kitti/datasets/
    # mulran: /mnt/088A6CBB8A6CA742/Datasets/MulRan/
    # apollo:
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--min_displacement', type=float, default=3.0)
    # Ignore query elements that do not have a corresponding map element within the given threshold (in meters)
    parser.add_argument('--dist_threshold', type=float, default=5.0)
    args = parser.parse_args()

    # Sequences are fixed
    sequence_name = '09'
    print(f'Dataset root: {args.dataset_root}')
    print(f'Kitti sequence: {sequence_name}')
    print(f'Minimum displacement between consecutive anchors: {args.min_displacement}')
    print(f'Ignore query elements without a corresponding map element within a threshold [m]: {args.dist_threshold}')

    # map / query
    sequence = Kitti360Sequence(args.dataset_root, sequence_name)
    eval_set = generate_evaluation_set(sequence, args.min_displacement, args.dist_threshold)
    create_seq_data(args.dataset_root, f'TestDataSeq09/map_seq1', f'pointcloud_{args.min_displacement}m', eval_set.map_set)
    create_seq_data(args.dataset_root, f'TestDataSeq09/query_seq1', f'pointcloud_{args.min_displacement}m', eval_set.query_set)