# Test sets for Mulran datasets.
# This file is adapted from: https://github.com/jac99/Egonn/blob/main/datasets/mulran/generate_evaluation_sets.py

import argparse
from typing import List
import os
import sys
import csv
import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


from datasets.mulran.mulran_raw import MulranSequence, MulranPointCloudLoader
from datasets.base_datasets import EvaluationTuple, EvaluationSet, filter_query_elements

# KAIST 02
MAP_TIMERANGE = (1566535940856033867, 1566536300000000000)
QUERY_TIMERANGE = (1566536300000000000, 1566536825534173166)
# Riverside 01:
# MAP_TIMERANGE = (1564718063503232284, 1564718300000000000)
# QUERY_TIMERANGE = (1564718300000000000, 1564718603800415528)


def get_scans(sequence: MulranSequence, ts_range: tuple = None) -> List[EvaluationTuple]:
    # Get a list of all readings from the test area in the sequence
    elems = []
    for ndx in range(len(sequence)):
        if ts_range is not None:
            if (ts_range[0] > sequence.timestamps[ndx]) or (ts_range[1] < sequence.timestamps[ndx]):
                continue
        pose = sequence.poses[ndx]
        position = pose[:2, 3]
        item = EvaluationTuple(sequence.timestamps[ndx], sequence.rel_scan_filepath[ndx], position=position, pose=pose)
        elems.append(item)
    return elems


def generate_evaluation_set(map_sequence: MulranSequence, query_sequence: MulranSequence,
                            dist_threshold=20) -> EvaluationSet:

    if map_sequence.sequence_name == query_sequence.sequence_name:
        map_set = get_scans(map_sequence, MAP_TIMERANGE)
        query_set = get_scans(query_sequence, QUERY_TIMERANGE)
    else:
        map_set = get_scans(map_sequence)
        query_set = get_scans(query_sequence)

    # Function used in evaluation datasets generation
    # Filters out query elements without a corresponding map element within dist_threshold threshold
    query_set = filter_query_elements(query_set, map_set, dist_threshold)
    print(f'{len(map_set)} database elements, {len(query_set)} query elements')
    return EvaluationSet(query_set, map_set)


# create map / query sequence data, refer to Oxford RobotCar / 3-Inhouse Dataset in PointNetVLAD
def create_seq_data(dataset_root: str, seq_name: str, scan_type: str, mq_set: List[EvaluationTuple]):
    print('Create Map/Query Sequence Data just like Oxford RobotCar Dataset in PointNetVLAD')
    pc_loader = MulranPointCloudLoader()
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
            writer.writerow([timestamp, anchor_pos[1], anchor_pos[0]])
            # save point cloud to *.bin file
            pc_file = os.path.join(scan_dir, timestamp + '.bin')
            anchor_pc.tofile(pc_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate evaluation sets for Mulran datasets')
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--sequence', type=str, required=True)
    parser.add_argument('--min_displacement', type=float, default=0.2)
    # Ignore query elements that do not have a corresponding map element within the given threshold (in meters)
    parser.add_argument('--dist_threshold', type=float, default=20)
    args = parser.parse_args()

    # Sequences is a list of (map sequence, query sequence)
    if args.sequence == 'DCC':
        sequences = [('DCC01', 'DCC02')]
        args.min_displacement = 10.0
        args.dist_threshold = 5
    elif args.sequence == 'Sejong':
        sequences = [('Sejong01', 'Sejong02')]
        args.min_displacement = 0.2
        args.dist_threshold = 20
    print(f'Dataset root: {args.dataset_root}')
    print(f'Minimum displacement between consecutive anchors: {args.min_displacement}')
    print(f'Ignore query elements without a corresponding map element within a threshold [m]: {args.dist_threshold}')

    count = 0
    split = 'test'
    for map_seq_name, query_seq_name in sequences:
        count += 1
        print(f'Map sequence: {map_seq_name}')
        print(f'Query sequence: {query_seq_name}')
        # map / query
        map_sequence = MulranSequence(args.dataset_root, map_seq_name, split=split, min_displacement=args.min_displacement)
        query_sequence = MulranSequence(args.dataset_root, query_seq_name, split=split, min_displacement=args.min_displacement)
        eval_set = generate_evaluation_set(map_sequence, query_sequence, dist_threshold=args.dist_threshold)
        create_seq_data(args.dataset_root, f'TestDataDCC01_02/map_seq{count}', f'pointcloud_{args.min_displacement}m', eval_set.map_set)
        create_seq_data(args.dataset_root, f'TestDataDCC01_02/query_seq{count}', f'pointcloud_{args.min_displacement}m', eval_set.query_set)
