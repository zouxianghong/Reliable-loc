import os.path

from collections import deque
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
import open3d

from utils.loading_pointclouds import *
from utils.util import compute_overlap_ratio, timestamp2str, check_makedirs
from datasets.dataset_info import dataset_info_dict
import datasets.query_pos_neg_dataset_pb2 as query_pos_neg_dataset_pb2


def mycollate(item):
    return item


class QueryPosNegTuple:
    """ one tuple: query, positives, negatives"""

    def __init__(self):
        self.positive_indices = []  # indices of positives in reference trip, list of query_idx_in_dataset
        self.negative_indices = []  # indices of positives in reference trip, list of query_idx_in_dataset


class SceneDataSet:
    """ one scene data """

    def __init__(self, name, for_training):
        # data config, see utils.dataset_info
        self.name = name
        self.data_cfg = dataset_info_dict[name].train_cfg() if for_training else dataset_info_dict[name].test_cfg()
        self.submap_type = self.data_cfg['submap_type']
        self.dataset_type = "train_" + self.submap_type
        if self.data_cfg['is_test_dataset']:
            self.dataset_type = "test_" + self.submap_type
        self.trip_names = []
        self.records = pd.DataFrame(columns=['file', 'northing', 'easting'])  # centroids: [DataRecord]
        self.sample_indices = []
        self.current_ref_trip_idx, self.positions, self.position_tree = -1, [], None  # positions of current ref trip
        self.records_size_list = []  # num of records in each trip
        self.valid_indices_in_dataset_list = []  # valid indices in datasets in each trip
        self.global_desc_list = []  # global descriptors produced by place recognition model
        # tuples within trips, Key: query_trip_idx, ref_trip_idx, Value: dict of QueryPosNegTuple(
        # Key: query_idx_in_dataset, Value:QueryPosNegTuple)
        self.query_pos_neg_tuples_dict = dict()
        self.query_trip_indices_load = []
        # hard negatives, Key: query_idx_in_dataset, Value: list of hard negative indices
        self.hard_negative_indices_dict = dict()
        # cached global / local features and positions
        self.cache_size = 1000
        self.pc_cache_idxs, self.fpfh_cache_idxs = deque(), deque()
        self.pc_dict, self.norm_meta_dict, self.fpfh_dict = dict(), dict(), dict()
        self.g_cache_idxs, self.l_cache_idxs = deque(), deque()
        self.g_desc_dict, self.l_kpt_dict, self.l_desc_dict = dict(), dict(), dict()

    def __len__(self):
        return len(self.records)

    def for_training(self):
        return not self.data_cfg['is_test_dataset']

    def set_cache_size(self, size):
        self.cache_size = size

    def print_stat_info(self):
        self.load(query_trip_indices=None)
        # num of query / map submaps / #pos per query
        n_trip_pair, n_query, n_map, n_pos = 0, 0, 0, 0
        sample_indices = self.get_indices_in_dataset()
        for ref_trip_idx in tqdm(range(len(self.trip_names)), desc="Stat Dataset info"):
            map_indices_in_dataset = sample_indices[ref_trip_idx]
            for query_trip_idx in range(len(self.trip_names)):
                if self.data_cfg['test_query_trips'] is not None and (
                        self.trip_names[query_trip_idx] not in self.data_cfg['test_query_trips']):
                    continue
                if self.data_cfg['is_test_dataset']:
                    if query_trip_idx == ref_trip_idx:
                        continue
                n_trip_pair += 1
                n_map += len(map_indices_in_dataset)
                query_indices_in_dataset = sample_indices[query_trip_idx]
                for i in tqdm(range(len(query_indices_in_dataset))):
                    query_idx_in_dataset = query_indices_in_dataset[i]
                    select_tuple = self.get_tuple(query_idx_in_dataset, ref_trip_idx,
                                                  skip_trip_itself=self.data_cfg['is_test_dataset'])
                    true_positives = select_tuple.positive_indices
                    if true_positives:
                        n_query += 1
                        n_pos += len(true_positives)
        if n_query:
            n_pos /= n_query
            n_query /= n_trip_pair
            n_map /= n_trip_pair
        print(f'n_trip_pair: {n_trip_pair}, n_query: {n_query}, n_map: {n_map}, n_pos_per_query: {n_pos}')

    def data_dir(self):
        return self.data_cfg['data_dir']

    def pickle_dir(self):
        if self.data_cfg['submap_type_ext'] is None:
            return os.path.join(self.data_dir(), "pickle_data")
        else:
            submap_type_ext = self.data_cfg['submap_type_ext']
            return os.path.join(self.data_dir(), f"pickle_data_{submap_type_ext}")

    def desc_dir(self, pr_backbone):
        basename_extra = 'test' if self.data_cfg['is_test_dataset'] else 'train'
        return os.path.join(self.pickle_dir(), f'desc_{pr_backbone}_{basename_extra}')

    def g_desc_dir(self, pr_backbone):
        return os.path.join(self.desc_dir(pr_backbone), 'global')

    def l_desc_dir(self, pr_backbone):
        return os.path.join(self.desc_dir(pr_backbone), 'local')

    def euc_knn_dir(self):
        return os.path.join(self.pickle_dir(), f'euc_knn')

    def set_current_ref_trip_idx(self, trip_idx):
        self.current_ref_trip_idx = trip_idx

    def get_trip_idx_by_name(self, name):
        for i in range(len(self.trip_names)):
            if self.trip_names[i] == name:
                return i
        return -1

    def get_indices_in_dataset(self, check_validation_pc=True):
        sum_records_size = 0
        self.sample_indices = []
        for trip_idx in range(len(self.trip_names)):
            records_size = self.records_size_list[trip_idx]
            indices = list(np.arange(sum_records_size, sum_records_size + records_size))
            # check indices whether has corresponding pc files
            if check_validation_pc:
                valid_indices = []
                for idx in indices:
                    pc_file = self.records.loc[idx]['file']
                    if os.path.exists(pc_file):
                        valid_indices.append(idx)
                indices = valid_indices
            self.sample_indices.append(indices)
            sum_records_size += records_size
        return self.sample_indices

    def get_hard_negative_indices(self, query_idx_in_dataset):
        hard_negative_indices = []
        if query_idx_in_dataset in self.hard_negative_indices_dict:
            hard_negative_indices = self.hard_negative_indices_dict[query_idx_in_dataset]
        return hard_negative_indices

    def get_query_idx_in_trip(self, query_idx_in_dataset):
        sum_records_size = 0
        for trip_idx in range(len(self.trip_names)):
            records_size = self.records_size_list[trip_idx]
            if sum_records_size <= query_idx_in_dataset < sum_records_size + records_size:
                query_idx_in_trip = query_idx_in_dataset - sum_records_size
                return trip_idx, query_idx_in_trip
            sum_records_size += records_size
        return -1, -1

    def get_query_idx_in_dataset(self, query_trip_idx, query_idx_in_trip):
        query_idx_in_dataset = 0
        for trip_idx in range(query_trip_idx):
            records_size = self.records_size_list[trip_idx]
            query_idx_in_dataset += records_size
        query_idx_in_dataset += query_idx_in_trip
        return query_idx_in_dataset

    def get_tuple(self, query_idx_in_dataset, ref_trip_idx=-1, skip_trip_itself=False):
        result_tuple = QueryPosNegTuple()
        query_trip_idx, query_idx_in_trip = self.get_query_idx_in_trip(query_idx_in_dataset)
        self.load([query_trip_idx], skip_trip_itself=skip_trip_itself)
        if ref_trip_idx == -1:
            for trip_idx in range(len(self.trip_names)):
                if query_trip_idx == trip_idx and skip_trip_itself:
                    continue
                if (query_trip_idx, trip_idx) not in self.query_pos_neg_tuples_dict:
                    continue
                if query_idx_in_dataset not in self.query_pos_neg_tuples_dict[query_trip_idx, trip_idx]:
                    continue
                select_tuple = self.query_pos_neg_tuples_dict[query_trip_idx, trip_idx][query_idx_in_dataset]
                result_tuple.positive_indices += select_tuple.positive_indices
                result_tuple.negative_indices += select_tuple.negative_indices
            return result_tuple
        else:
            if query_trip_idx == ref_trip_idx and skip_trip_itself:
                return result_tuple
            if (query_trip_idx, ref_trip_idx) not in self.query_pos_neg_tuples_dict:
                return result_tuple
            if query_idx_in_dataset not in self.query_pos_neg_tuples_dict[query_trip_idx, ref_trip_idx]:
                return result_tuple
            return self.query_pos_neg_tuples_dict[query_trip_idx, ref_trip_idx][query_idx_in_dataset]

    def get_pos_xy(self, idx, with_offset=True):
        g_offset = self.data_cfg['global_offset']
        position = self.records.loc[idx]
        if with_offset:
            position = np.array([position['easting'] - g_offset[0,0], position['northing'] - g_offset[0,1]], dtype=np.float32)
        else:
            position = np.array([position['easting'], position['northing']], dtype=np.float32)
        return position

    def get_pos_xyz(self, idx, with_offset=True):
        g_offset = self.data_cfg['global_offset']
        position = self.records.loc[idx]
        if with_offset:
            position = np.array([position['easting']-g_offset[0,0], position['northing']-g_offset[0,1], 0.0], dtype=np.float32)
        else:
            position = np.array([position['easting'], position['northing'], 0.0], dtype=np.float32)
        return position

    def get_dist(self, idx1, idx2):
        p1 = self.get_pos_xy(idx1)
        p2 = self.get_pos_xy(idx2)
        dist = np.linalg.norm(p1 - p2)
        return dist

    def get_label(self, idx1, idx2):
        """ pos: 1, neg: 0, unknown: -1 """
        dist = self.get_dist(idx1, idx2)
        if self.data_cfg['is_test_dataset']:  # for testing
            label = 1 if dist < self.data_cfg['search_radius_pos'] else 0
        else:  # for training
            if dist < self.data_cfg['search_radius_pos']:
                label = 1
            elif dist > self.data_cfg['search_radius_neg']:
                label = 0
            else:  # unknown
                label = -1
        return label

    def get_pc_files(self, idxs):
        idxs = list(idxs)
        files = []
        for idx in idxs:
            files.append(self.records.loc[idx]['file'])
        return files

    def last_or_next(self, idx, step=-1):  # make sure record sorted in increasing order
        last_trip_idx, last_idx_in_trip = self.get_query_idx_in_trip(idx + step)
        if last_trip_idx == -1:
            return -1
        current_trip_idx, current_idx_in_trip = self.get_query_idx_in_trip(idx)
        if last_trip_idx != current_trip_idx:
            return -1
        last_record, current_record = self.records.loc[idx+step], self.records.loc[idx]
        test_region_vertices = self.data_cfg['test_region_vertices']
        test_region_width = self.data_cfg['test_region_width']
        last_in_test = self.__check_in_test_region(last_record['northing'], last_record['easting'],
                                                   test_region_vertices, test_region_width, test_region_width)
        current_in_test = self.__check_in_test_region(last_record['northing'], last_record['easting'],
                                                      test_region_vertices, test_region_width, test_region_width)
        lc_xor = bool(last_in_test ^ current_in_test)
        if lc_xor:
            return -1
        return idx+step

    def last_k(self, idx, k=3, interval=3):
        last_k_idxs = []
        for step in range(-k*interval, 0, interval):
            last_idx = self.last_or_next(idx, step)
            if last_idx == -1:
                break
            last_k_idxs.append(last_idx)
        return last_k_idxs

    def next_k(self, idx, k=3, interval=3):
        next_k_idxs = []
        for step in range(interval, (k+1) * interval, interval):
            next_idx = self.last_or_next(idx, step)
            if next_idx == -1:
                break
            next_k_idxs.append(next_idx)
        return next_k_idxs

    def construct_position_tree(self, idxs=None):
        if idxs is None:
            if self.current_ref_trip_idx == -1:
                idxs = list(range(len(self.records)))
            else:
                sample_indices = self.get_indices_in_dataset()
                idxs = sample_indices[self.current_ref_trip_idx]
        self.positions = [self.get_pos_xy(i) for i in idxs]
        self.positions = np.stack(self.positions, axis=0)  # * x 2
        self.position_tree = KDTree(self.positions)
        return self.position_tree

    def destroy_position_tree(self):
        self.position_tree = None

    def clear_global_descs(self):
        self.global_desc_list = []

    def clear_tuples(self, query_trip_indices=None):
        if isinstance(query_trip_indices, list):
            for query_trip_idx in query_trip_indices:
                for ref_trip_idx in range(len(self.trip_names)):
                    if (query_trip_idx, ref_trip_idx) in self.query_pos_neg_tuples_dict:
                        self.query_pos_neg_tuples_dict.pop((query_trip_idx, ref_trip_idx))
                self.query_trip_indices_load.remove(query_trip_idx)
        else:
            self.query_pos_neg_tuples_dict = dict()
            self.query_trip_indices_load = []

    def get_overlap_indices(self, query_idx_in_dataset, positive_indices):
        nn_dict = dict()
        # load {dataset_type}_overlap_indices_{query_idx_in_dataset}.pb
        pickle_dir = self.pickle_dir()
        pb_file = os.path.join(pickle_dir, "{}_overlap_indices_{}.pb".format(self.dataset_type, query_idx_in_dataset))
        if not os.path.exists(pb_file):
            return None
        # print("load  from pb file: ", pb_file)
        overlap_indices_pb = query_pos_neg_dataset_pb2.QueryOverlapIndices()
        with open(pb_file, 'rb') as bf:
            binary_data = bf.read()
            overlap_indices_pb.ParseFromString(binary_data)
        for qp_overlap_indices_pb in overlap_indices_pb.qp_overlap_indices:
            nn_dict[query_idx_in_dataset, qp_overlap_indices_pb.positive_idx] = qp_overlap_indices_pb.overlap_indices

        overlap_indices = dict()
        for i in range(len(positive_indices)):
            pos_idx = positive_indices[i]
            overlap_indices[0, i + 1] = nn_dict[query_idx_in_dataset, pos_idx]
        return overlap_indices

    def create(self, cloud_ext='.bin', trip_names=None, test_region_vertices=None,
               test_region_width=50.0, search_radius_pos=25.0, search_radius_neg=50.0, check_overlap=False,
               overlap_thresh=0.1, max_neg=15000):
        if test_region_vertices is None:
            test_region_vertices = []
        # get trip names
        self.trip_names = []
        trip_dirs = sorted(os.listdir(self.data_dir()))
        if isinstance(trip_names, list):
            for trip_dir in trip_names:
                if 'pickle_data' not in trip_dir and trip_dir in trip_dirs:
                    self.trip_names.append(trip_dir)
        else:
            for trip_dir in trip_dirs:
                if not os.path.isdir(os.path.join(self.data_dir(), trip_dir)):
                    continue
                if 'pickle_data' not in trip_dir:
                    self.trip_names.append(trip_dir)
        print("create datasets: {}, trips: {}".format(self.data_dir(), self.trip_names))
        # get records (centroids)
        centroid_csv_name = self.submap_type + ".csv"
        for trip_dir in self.trip_names:
            centroid_csv = os.path.join(self.data_dir(), trip_dir, centroid_csv_name)
            trip_records = pd.read_csv(centroid_csv, sep=',')
            trip_records = trip_records.sort_values('timestamp')  # sort by time stamp
            trip_records.reset_index(drop=True)
            cloud_dir = os.path.join(self.data_dir(), trip_dir, self.submap_type)
            for index, row in trip_records.iterrows():
                trip_records.loc[index, 'timestamp'] = os.path.join(cloud_dir, timestamp2str(
                    trip_records.loc[index, 'timestamp']) + cloud_ext)
            trip_records = trip_records.rename(columns={'timestamp': 'file'})
            if self.data_cfg['is_test_dataset'] is False:
                df_train = pd.DataFrame(columns=['file', 'northing', 'easting'])
                for index, row in trip_records.iterrows():
                    if (self.__check_in_test_region(row['northing'], row['easting'], test_region_vertices,
                                                    test_region_width, test_region_width)):
                        continue
                    else:
                        df_train = df_train.append(row, ignore_index=True)
                trip_records = df_train
            self.records = pd.concat([self.records, trip_records], ignore_index=True)
            self.records_size_list.append(len(trip_records))
            self.valid_indices_in_dataset_list.append(set())
        if len(self.records) == 0:
            return None
        # ensure dir to save *.pickle files
        pickle_dir = self.pickle_dir()
        if not os.path.exists(pickle_dir):
            os.mkdir(pickle_dir)
        # save query_pos_neg_tuples_dict to *.pkl
        database_tree = KDTree(self.records[['northing', 'easting']])
        for query_trip_idx in range(len(self.trip_names)):
            records_size = self.records_size_list[query_trip_idx]
            for ref_trip_idx in range(len(self.trip_names)):
                self.query_pos_neg_tuples_dict[query_trip_idx, ref_trip_idx] = dict()
            progress_desc = "Find Positives and Negatives in {}".format(self.trip_names[query_trip_idx])
            for query_idx_in_trip in tqdm(range(records_size), desc=progress_desc):
                query_idx_in_dataset = self.get_query_idx_in_dataset(query_trip_idx, query_idx_in_trip)
                query_record = self.records.loc[query_idx_in_dataset]
                # check whether query is in test region
                in_test_region = self.__check_in_test_region(query_record["northing"], query_record["easting"],
                                                             test_region_vertices, test_region_width, test_region_width)
                # skip if: is_test_dataset == True and in_test_region == False
                # skip if: is_test_dataset == False and in_test_region == True
                if self.data_cfg['is_test_dataset'] ^ in_test_region:
                    continue
                # find positives
                positive_indices = []
                negative_indices = []
                query_centroid = np.array([[query_record["northing"], query_record["easting"]]])
                index = database_tree.query_radius(query_centroid, r=search_radius_pos)
                found_indices = np.setdiff1d(index[0], [query_idx_in_dataset]).tolist()
                if check_overlap:
                    query_pc = np.load(query_record["file"])
                    for ref_idx_in_dataset in found_indices:
                        ref_record = self.records.loc[query_idx_in_dataset]
                        ref_pc = np.load(ref_record["file"])
                        overlap_ratio = compute_overlap_ratio(query_pc, ref_pc, use_2d=True, use_grid=True)
                        if overlap_ratio > overlap_thresh:
                            positive_indices.append(ref_idx_in_dataset)
                        elif overlap_ratio < 0.01:
                            negative_indices.append(ref_idx_in_dataset)
                else:
                    positive_indices = found_indices
                # find negatives
                index = database_tree.query_radius(query_centroid, r=search_radius_neg)
                negative_indices += np.setdiff1d(self.records.index.values.tolist(), index[0]).tolist()
                if len(negative_indices) > max_neg:
                    negative_indices = random.sample(negative_indices, max_neg)
                # put positives and negatives into tuple dict
                for idx_in_dataset in positive_indices:
                    ref_trip_idx, idx_in_trip = self.get_query_idx_in_trip(idx_in_dataset)
                    if query_idx_in_dataset not in self.query_pos_neg_tuples_dict[query_trip_idx, ref_trip_idx]:
                        self.query_pos_neg_tuples_dict[query_trip_idx, ref_trip_idx][
                            query_idx_in_dataset] = QueryPosNegTuple()
                    query_pos_neg_tuple = self.query_pos_neg_tuples_dict[query_trip_idx, ref_trip_idx][
                        query_idx_in_dataset]
                    query_pos_neg_tuple.positive_indices.append(idx_in_dataset)
                for idx_in_dataset in negative_indices:
                    ref_trip_idx, idx_in_trip = self.get_query_idx_in_trip(idx_in_dataset)
                    if query_idx_in_dataset not in self.query_pos_neg_tuples_dict[query_trip_idx, ref_trip_idx]:
                        self.query_pos_neg_tuples_dict[query_trip_idx, ref_trip_idx][
                            query_idx_in_dataset] = QueryPosNegTuple()
                    query_pos_neg_tuple = self.query_pos_neg_tuples_dict[query_trip_idx, ref_trip_idx][
                        query_idx_in_dataset]
                    query_pos_neg_tuple.negative_indices.append(idx_in_dataset)
                # add query_idx_in_dataset into valid_indices_in_dataset_list
                if positive_indices and negative_indices:
                    self.valid_indices_in_dataset_list[query_trip_idx].add(query_idx_in_dataset)
            # save query_pos_neg_tuples_dict to *.pkl
            for ref_trip_idx in range(len(self.trip_names)):
                tuples_pkl = "{}_tuples_{}_to_{}.pickle".format(self.dataset_type, query_trip_idx, ref_trip_idx)
                tuples_pkl = os.path.join(pickle_dir, tuples_pkl)
                if len(self.query_pos_neg_tuples_dict[query_trip_idx, ref_trip_idx]) > 0:
                    with open(tuples_pkl, 'wb') as handle:
                        pickle.dump(self.query_pos_neg_tuples_dict[query_trip_idx, ref_trip_idx], handle,
                                    protocol=pickle.HIGHEST_PROTOCOL)
                        self.query_pos_neg_tuples_dict.pop((query_trip_idx, ref_trip_idx))  # release memory
            tuples_pkl = "{}_tuples_{}_to_{}.pickle".format(self.dataset_type, query_trip_idx,
                                                            "[{}~{}]".format(0, len(self.trip_names) - 1))
            tuples_pkl = os.path.join(pickle_dir, tuples_pkl)
            print("save: ", tuples_pkl)
        # save records and record size of each trip to *.pkl
        records_pkl = os.path.join(pickle_dir, "{}_records.pickle".format(self.dataset_type))
        with open(records_pkl, 'wb') as handle:
            pickle.dump((self.trip_names, self.records, self.records_size_list, self.valid_indices_in_dataset_list),
                        handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("save: ", records_pkl)

    def create_from_pb(self, max_neg=15000):
        # load pb
        if self.data_cfg['is_test_dataset']:
            pb_file = "test_"
        else:
            pb_file = "train_"
        pb_file += self.submap_type + "_dataset.pb"
        pb_file = os.path.join(self.pickle_dir(), pb_file)
        print("create dataset from pb file: ", pb_file)
        dataset_pb = query_pos_neg_dataset_pb2.DataSet()
        with open(pb_file, 'rb') as bf:
            binary_data = bf.read()
            dataset_pb.ParseFromString(binary_data)
        # convert pb message to pickle
        for trip_name in dataset_pb.trip_names:
            self.trip_names.append(trip_name)
        for record in dataset_pb.records:
            new_row = {'file': record.name, 'northing': record.x, 'easting': record.y}
            self.records = self.records._append(new_row, ignore_index=True)
        record_arange = list(np.arange(0, len(self.records)))
        for record_size in dataset_pb.records_size_list:
            self.records_size_list.append(record_size)
            self.valid_indices_in_dataset_list.append(set())
        query_indices_in_dataset = []
        for query_idx_in_dataset in dataset_pb.tuples:
            query_indices_in_dataset.append(query_idx_in_dataset)
        for i in tqdm(range(len(query_indices_in_dataset)), desc="Create Dataset from pb file"):
            query_idx_in_dataset = query_indices_in_dataset[i]
            query_trip_idx, idx_in_query_trip = self.get_query_idx_in_trip(query_idx_in_dataset)
            select_tuple = dataset_pb.tuples[query_idx_in_dataset]
            for idx_in_dataset_pos in select_tuple.positive_indices:
                ref_trip_idx, idx_in_ref_trip = self.get_query_idx_in_trip(idx_in_dataset_pos)
                if (query_trip_idx, ref_trip_idx) not in self.query_pos_neg_tuples_dict:
                    self.query_pos_neg_tuples_dict[query_trip_idx, ref_trip_idx] = dict()
                if query_idx_in_dataset not in self.query_pos_neg_tuples_dict[query_trip_idx, ref_trip_idx]:
                    self.query_pos_neg_tuples_dict[query_trip_idx, ref_trip_idx][query_idx_in_dataset] = QueryPosNegTuple()
                query_pos_neg_tuple = self.query_pos_neg_tuples_dict[query_trip_idx, ref_trip_idx][query_idx_in_dataset]
                query_pos_neg_tuple.positive_indices.append(idx_in_dataset_pos)
            non_negative_indices = []
            for idx_in_dataset_non_neg in select_tuple.non_negative_indices:
                non_negative_indices.append(idx_in_dataset_non_neg)
            negative_indices = np.setdiff1d(record_arange, non_negative_indices).tolist()
            if len(negative_indices) > max_neg:
                    negative_indices = random.sample(negative_indices, max_neg)
            for idx_in_dataset_neg in negative_indices:
                ref_trip_idx, idx_in_ref_trip = self.get_query_idx_in_trip(idx_in_dataset_neg)
                if (query_trip_idx, ref_trip_idx) not in self.query_pos_neg_tuples_dict:
                    self.query_pos_neg_tuples_dict[query_trip_idx, ref_trip_idx] = dict()
                if query_idx_in_dataset not in self.query_pos_neg_tuples_dict[query_trip_idx, ref_trip_idx]:
                    self.query_pos_neg_tuples_dict[query_trip_idx, ref_trip_idx][query_idx_in_dataset] = QueryPosNegTuple()
                query_pos_neg_tuple = self.query_pos_neg_tuples_dict[query_trip_idx, ref_trip_idx][query_idx_in_dataset]
                query_pos_neg_tuple.negative_indices.append(idx_in_dataset_neg)
            if select_tuple.positive_indices and negative_indices:
                self.valid_indices_in_dataset_list[query_trip_idx].add(query_idx_in_dataset)
            dataset_pb.tuples.pop(query_idx_in_dataset)
        # ensure dir to save *.pickle files
        pickle_dir = self.pickle_dir()
        if not os.path.exists(pickle_dir):
            os.mkdir(pickle_dir)
        # save query_pos_neg_tuples_dict to *.pkl
        for query_trip_idx in range(len(self.trip_names)):
            for ref_trip_idx in range(len(self.trip_names)):
                tuples_pkl = "{}_tuples_{}_to_{}.pickle".format(self.dataset_type, query_trip_idx, ref_trip_idx)
                tuples_pkl = os.path.join(pickle_dir, tuples_pkl)
                if (query_trip_idx, ref_trip_idx) in self.query_pos_neg_tuples_dict:
                    with open(tuples_pkl, 'wb') as handle:
                        pickle.dump(self.query_pos_neg_tuples_dict[query_trip_idx, ref_trip_idx], handle,
                                    protocol=pickle.HIGHEST_PROTOCOL)
                        self.query_pos_neg_tuples_dict.pop((query_trip_idx, ref_trip_idx))  # release memory
            tuples_pkl = "{}_tuples_{}_to_{}.pickle".format(self.dataset_type, query_trip_idx,
                                                            "[{}~{}]".format(0, len(self.trip_names) - 1))
            tuples_pkl = os.path.join(pickle_dir, tuples_pkl)
            print("save: ", tuples_pkl)
        # save records and record size of each trip to *.pkl
        records_pkl = os.path.join(pickle_dir, "{}_records.pickle".format(self.dataset_type))
        with open(records_pkl, 'wb') as handle:
            pickle.dump((self.trip_names, self.records, self.records_size_list, self.valid_indices_in_dataset_list),
                        handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("save: ", records_pkl)
    
    def load(self, query_trip_indices=None, skip_trip_itself=False):
        # load trip data
        pickle_dir = self.pickle_dir()
        if len(self.records) == 0:
            records_pkl = os.path.join(pickle_dir, "{}_records.pickle".format(self.dataset_type))
            if os.path.exists(records_pkl):
                with open(records_pkl, 'rb') as handle:
                    self.trip_names, self.records, self.records_size_list, self.valid_indices_in_dataset_list = pickle.load(
                        handle)
                print("load: ", records_pkl)
            else:
                print("file not exist: ", records_pkl)
                return
        # load query/positive/negative tuples
        if query_trip_indices == -1:
            query_trip_indices = list(range(len(self.trip_names)))
        if isinstance(query_trip_indices, list):
            for query_trip_idx in query_trip_indices:
                load = False
                for ref_trip_idx in range(len(self.trip_names)):
                    if query_trip_idx == ref_trip_idx and skip_trip_itself:
                        continue
                    if (query_trip_idx, ref_trip_idx) in self.query_pos_neg_tuples_dict:
                        continue
                    load = True
                    self.__load_one_tuple_pkl(query_trip_idx, ref_trip_idx)
                if load:
                    if query_trip_idx in self.query_trip_indices_load:
                        self.query_trip_indices_load.remove(query_trip_idx)
                    self.query_trip_indices_load.append(query_trip_idx)
                    pickle_dir = self.pickle_dir()
                    tuples_pkl = "{}_tuples_{}_to_{}.pickle".format(self.dataset_type, query_trip_idx,
                                                                    "[{}~{}]".format(0, len(self.trip_names) - 1))
                    tuples_pkl = os.path.join(pickle_dir, tuples_pkl)
                    print("load: ", tuples_pkl)

    def __load_one_tuple_pkl(self, query_trip_idx, ref_trip_idx):
        pickle_dir = self.pickle_dir()
        tuples_pkl = "{}_tuples_{}_to_{}.pickle".format(self.dataset_type, query_trip_idx, ref_trip_idx)
        tuples_pkl = os.path.join(pickle_dir, tuples_pkl)
        if os.path.exists(tuples_pkl):
            with open(tuples_pkl, 'rb') as handle:
                query_pos_neg_tuples = pickle.load(handle)
                self.query_pos_neg_tuples_dict[query_trip_idx, ref_trip_idx] = query_pos_neg_tuples

    def get_query_pos_neg_tuple(self, query_idx_in_dataset, num_pos, num_neg, other_neg=False,
                                num_hard_neg=10, num_sample_neg=3000, normalize_cloud=True,
                                skip_trip_itself=False, load_overlap_indices=False):
        # hardest negative mining
        if len(self.global_desc_list) == 0:
            return self.__get_training_tuple(query_idx_in_dataset, num_pos, num_neg, other_neg, normalize_cloud,
                                             skip_trip_itself, load_overlap_indices)
        # hardest negative mining
        select_tuple = self.get_tuple(query_idx_in_dataset, -1, skip_trip_itself)
        if len(select_tuple.negative_indices) > num_sample_neg:
            negative_indices = np.random.choice(select_tuple.negative_indices, num_sample_neg, replace=False)
        else:
            negative_indices = select_tuple.negative_indices
        hard_negative_indices = self.get_hard_negative_indices(query_idx_in_dataset)
        if hard_negative_indices:
            negative_indices = np.concatenate([negative_indices, hard_negative_indices], axis=0)
        self.hard_negative_indices_dict[query_idx_in_dataset] = self.__get_hard_negatives(
            self.global_desc_list[query_idx_in_dataset], self.global_desc_list, negative_indices, num_hard_neg)
        return self.__get_training_tuple(query_idx_in_dataset, num_pos, num_neg, other_neg, normalize_cloud,
                                         skip_trip_itself, load_overlap_indices)

    def make_descs(self, model, cfg, device, batch_size=20, stat_time=False, save=False):
        # save dirs
        if save:
            g_desc_dir = self.g_desc_dir(cfg['model_type'])
            check_makedirs(g_desc_dir)
            l_desc_dir = self.l_desc_dir(cfg['model_type'])
            check_makedirs(l_desc_dir)
        if cfg['model_type'] == 'minkloc3d_v2':
            from place_recognition.Minkloc3D_V2.misc.utils import TrainingParams
            params = TrainingParams(cfg['config'], cfg['model_config'])
        elif cfg['model_type'] == 'egonn':
            from place_recognition.EgoNN.misc.utils import TrainingParams
            params = TrainingParams(cfg['config'], cfg['model_config'])
        model.eval()
        run_time = []
        self.clear_global_descs()
        for i in tqdm(range(len(self.records) // batch_size + 1), desc="Make Global Descriptors"):
            beg_idx = i * batch_size
            if beg_idx == len(self.records):
                break
            end_idx = (i + 1) * batch_size
            if end_idx > len(self.records):
                end_idx = len(self.records)
            normalize, zoom = False, False
            if self.data_cfg['self_collected']:
                normalize, zoom = True, True
                if cfg['model_type'] == 'egonn' or cfg['model_type'] == 'lcdnet' or cfg['model_type'] == 'logg3d_net':
                    zoom = False
            selected_indices = list(range(beg_idx, end_idx))
            pcs, norm_metas = self.get_pcs(selected_indices, normalize=normalize, zoom=zoom, return_norm_meta=True)
            if cfg['model_type'] == 'minkloc3d_v2':
                import MinkowskiEngine as ME
                with torch.no_grad():
                    for b_i in range(len(pcs)):
                        if pcs[b_i] is None:
                            self.global_desc_list.append(None)
                            continue
                        coords, _ = params.model_params.quantizer(pcs[b_i])
                        bcoords = ME.utils.batched_coordinates([coords])
                        feats = torch.ones((bcoords.shape[0], 1), dtype=torch.float32)
                        batch = {'coords': bcoords.to(device), 'features': feats.to(device)}
                        if stat_time:
                            torch.cuda.synchronize()
                        start_t = time.time()
                        g_desc = model(batch).detach().cpu().numpy()
                        if stat_time:
                            torch.cuda.synchronize()
                        end_t = time.time()
                        run_time_i = (end_t - start_t) * 1000  # unit: ms
                        run_time.append(run_time_i)
                        g_desc = g_desc.reshape(-1, g_desc.shape[-1])  # 1 x C
                        self.global_desc_list.append(g_desc)
                        # save global desc
                        if save:
                            g_desc_pkl = os.path.join(g_desc_dir, f"{selected_indices[b_i]}.pickle")
                            with open(g_desc_pkl, 'wb') as handle:
                                pickle.dump(g_desc, handle, protocol=pickle.HIGHEST_PROTOCOL)
                            # TODO: local desc?
            elif cfg['model_type'] == 'egonn':
                import MinkowskiEngine as ME
                with torch.no_grad():
                    for b_i in range(len(pcs)):
                        if pcs[b_i] is None:
                            self.global_desc_list.append(None)
                            continue
                        pc = torch.from_numpy(pcs[b_i])
                        coords, _ = params.model_params.quantizer(pc)
                        bcoords = ME.utils.batched_coordinates([coords])
                        feats = torch.ones((bcoords.shape[0], 1), dtype=torch.float32)
                        batch = {'coords': bcoords.to(device), 'features': feats.to(device)}
                        if stat_time:
                            torch.cuda.synchronize()
                        start_t = time.time()
                        y = model(batch)
                        if stat_time:
                            torch.cuda.synchronize()
                        end_t = time.time()
                        run_time_i = (end_t - start_t) * 1000  # unit: ms
                        run_time.append(run_time_i)
                        g_desc = y['global'].detach().cpu().numpy()
                        g_desc = g_desc.reshape(-1, g_desc.shape[-1])  # 1 x C
                        self.global_desc_list.append(g_desc)
                        l_pos, l_desc, sigma = y['keypoints'][0], y['descriptors'][0], y['sigma'][0]
                        # Get n_k keypoints with the lowest uncertainty sorted in increasing order
                        _, ndx = torch.topk(sigma.squeeze(1), dim=0, k=128, largest=False)
                        l_pos = l_pos[ndx].detach().cpu().numpy()  # K x 3
                        l_desc = l_desc[ndx].detach().cpu().numpy()  # K x C
                        norm_meta = norm_metas[b_i]  # scale and translation
                        # save desc
                        if save:
                            # global desc
                            g_desc_pkl = os.path.join(g_desc_dir, f"{selected_indices[b_i]}.pickle")
                            with open(g_desc_pkl, 'wb') as handle:
                                pickle.dump(g_desc, handle, protocol=pickle.HIGHEST_PROTOCOL)
                            # local desc
                            l_desc_pkl = os.path.join(l_desc_dir, f"{selected_indices[b_i]}.pickle")
                            with open(l_desc_pkl, 'wb') as handle:
                                pickle.dump((l_pos, l_desc, norm_meta), handle, protocol=pickle.HIGHEST_PROTOCOL)
            elif cfg['model_type'] == 'lcdnet':
                from pcdet.datasets.kitti.kitti_dataset import KittiDataset
                with torch.no_grad():
                    for b_i in range(len(pcs)):
                        if pcs[b_i] is None:
                            self.global_desc_list.append(None)
                            continue
                        pc = pcs[b_i]
                        oo = np.ones(len(pc)).reshape((-1, 1))
                        pc = np.hstack((pc, oo)).astype(np.float32)
                        pc = torch.from_numpy(pc).to(device)
                        anchor_list = []
                        anchor_list.append(model.backbone.prepare_input(pc))
                        model_in = KittiDataset.collate_batch(anchor_list)
                        for key, val in model_in.items():
                            if not isinstance(val, np.ndarray):
                                continue
                            model_in[key] = torch.from_numpy(val).float().to(device)
                        if stat_time:
                            torch.cuda.synchronize()
                        start_t = time.time()
                        batch_dict = model(model_in, metric_head=False, compute_rotation=True, compute_transl=False)
                        if stat_time:
                            torch.cuda.synchronize()
                        end_t = time.time()
                        run_time_i = (end_t - start_t) * 1000  # unit: ms
                        run_time.append(run_time_i)
                        # global
                        output_desc = batch_dict['out_embedding']
                        g_desc = output_desc.cpu().detach().numpy()
                        g_desc = np.reshape(g_desc, (1, -1))
                        self.global_desc_list.append(g_desc)
                        # local
                        l_pos = batch_dict['point_coords'].view(batch_dict['batch_size'], -1, 4).clone()
                        l_desc = batch_dict['point_features'].squeeze(-1).squeeze().clone()
                        l_pos = l_pos.squeeze()[:, 1:].cpu().detach().numpy()
                        l_desc = l_desc.permute(1, 0).cpu().detach().numpy()
                        norm_meta = norm_metas[b_i]  # scale and translation
                        # save desc
                        if save:
                            # global desc
                            g_desc_pkl = os.path.join(g_desc_dir, f"{selected_indices[b_i]}.pickle")
                            with open(g_desc_pkl, 'wb') as handle:
                                pickle.dump(g_desc, handle, protocol=pickle.HIGHEST_PROTOCOL)
                            # local desc
                            l_desc_pkl = os.path.join(l_desc_dir, f"{selected_indices[b_i]}.pickle")
                            with open(l_desc_pkl, 'wb') as handle:
                                pickle.dump((l_pos, l_desc, norm_meta), handle, protocol=pickle.HIGHEST_PROTOCOL)
            elif cfg['model_type'] == 'logg3d_net':
                from place_recognition.LoGG3DNet.eval_util import make_sparse_tensor, compute_embedding
                with torch.no_grad():
                    for b_i in range(len(pcs)):
                        if pcs[b_i] is None:
                            self.global_desc_list.append(None)
                            continue
                        pc = pcs[b_i]
                        oo = np.ones(len(pc)).reshape((-1, 1))
                        xyzi = np.hstack((pc, oo)).astype(np.float32)
                        hash_vals, coords, indices, input, points = make_sparse_tensor(
                            xyzi, voxel_size=0.5, num_points=35000, return_points=True, return_hash=True)
                        coords = coords.astype(np.int32)
                        input = input.cuda()
                        points = points[:, :3]
                        if stat_time:
                            torch.cuda.synchronize()
                        start_t = time.time()
                        g_desc, l_pos, l_desc = compute_embedding(input, model, coords, points, is_dense=True)
                        if stat_time:
                            torch.cuda.synchronize()
                        end_t = time.time()
                        run_time_i = (end_t - start_t) * 1000  # unit: ms
                        run_time.append(run_time_i)
                        l_pos = l_pos.float().detach().cpu().numpy()
                        l_desc = l_desc.float().detach().cpu().numpy()
                        norm_meta = norm_metas[b_i]  # scale and translation
                        self.global_desc_list.append(g_desc)
                        # save desc
                        if save:
                            # global desc
                            g_desc_pkl = os.path.join(g_desc_dir, f"{selected_indices[b_i]}.pickle")
                            with open(g_desc_pkl, 'wb') as handle:
                                pickle.dump(g_desc, handle, protocol=pickle.HIGHEST_PROTOCOL)
                            # local desc
                            l_desc_pkl = os.path.join(l_desc_dir, f"{selected_indices[b_i]}.pickle")
                            with open(l_desc_pkl, 'wb') as handle:
                                pickle.dump((l_pos, l_desc, norm_meta), handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                valid_pcs, valid_selected_indices = [], []
                for b_i in range(len(pcs)):
                    if pcs[b_i] is not None:
                        valid_pcs.append(pcs[b_i])
                        valid_selected_indices.append(selected_indices[b_i])
                valid_pcs = np.array(valid_pcs)
                feed_tensor = torch.from_numpy(valid_pcs).float()
                feed_tensor = feed_tensor.unsqueeze(1)  # B x 1 x N x 3
                feed_tensor = feed_tensor.to(device, non_blocking=True)
                with torch.no_grad():
                    if stat_time:
                        torch.cuda.synchronize()
                    start_t = time.time()
                    global_descs, center_idx, local_descs = model(feed_tensor), None, None
                    if isinstance(global_descs, tuple):
                        global_descs, local_descs, center_idx = global_descs
                    global_descs = global_descs.detach().cpu().numpy()
                    global_descs = np.squeeze(global_descs)  # B x C
                    global_descs = global_descs.reshape([-1, global_descs.shape[-1]])
                    if stat_time:
                        torch.cuda.synchronize()
                    end_t = time.time()
                    run_time_i = (end_t - start_t) * 1000 / (end_idx - beg_idx)  # unit: ms
                    run_time_i = [run_time_i] * (end_idx - beg_idx)
                    run_time = run_time + run_time_i
                if save:
                    feed_tensor = feed_tensor.squeeze(1)  # B x N x 3
                    b = global_descs.shape[0]
                    if local_descs is not None:
                        local_descs = local_descs[-2].squeeze(-1).permute(0, 2, 1).detach().cpu().numpy()  # B x K x C
                        center_idx = center_idx[0]  # B x K
                    for b_i in range(b):
                        g_desc = global_descs[b_i]  # C
                        g_desc = g_desc.reshape(1, -1)  # 1 x C
                        g_desc_pkl = os.path.join(g_desc_dir, f"{valid_selected_indices[b_i]}.pickle")
                        with open(g_desc_pkl, 'wb') as handle:
                            pickle.dump(g_desc, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        if center_idx is not None:
                            l_pos = torch.index_select(feed_tensor[b_i].squeeze(), dim=0,
                                                       index=center_idx[b_i])  # K(=1024)
                            l_pos = l_pos.detach().cpu().numpy()  # K x 3
                            l_desc = local_descs[b_i]  # K x C
                            norm_meta = norm_metas[b_i]  # scale and translation
                            l_desc_pkl = os.path.join(l_desc_dir, f"{valid_selected_indices[b_i]}.pickle")
                            with open(l_desc_pkl, 'wb') as handle:
                                pickle.dump((l_pos, l_desc, norm_meta), handle, protocol=pickle.HIGHEST_PROTOCOL)
                count = -1
                for b_i in range(len(pcs)):
                    if pcs[b_i] is not None:
                        count += 1
                        g_desc = global_descs[count]
                        self.global_desc_list.append(g_desc.reshape(1, -1))
                    else:
                        self.global_desc_list.append(None)
        self.global_desc_list = np.concatenate(self.global_desc_list, axis=0)
        run_time_mean, run_time_std = np.mean(run_time), np.std(run_time)
        print('run time: {:.3f} +- {:.3f} ms'.format(run_time_mean, run_time_std))

    def get_pc(self, idx, downsample=False, normalize=False, zoom=True, return_norm_meta=False, use_real_center=False):
        """ Return: N x 3 np array"""
        if idx not in self.pc_dict:
            # load pc
            pc_file = self.records.loc[idx]['file']
            if os.path.exists(pc_file):
                pc = load_pc_file(pc_file, use_np_load=self.data_cfg['self_collected'], dtype=self.data_cfg['cloud_dtype'])
                if pc.shape[0] > 4096 and downsample:
                    sample_idxs = np.random.choice(pc.shape[0], 4096, replace=False)
                    pc = pc[sample_idxs]
                pc = pc - self.data_cfg['global_offset']
                norm_meta = {'scale': 1.0, 'trans': np.zeros([1, 3])}
                if normalize:
                    if return_norm_meta:
                        pc, norm_meta = normalize_point_cloud(pc, return_norm_meta, zoom)
                    else:
                        pc = normalize_point_cloud(pc, return_norm_meta, zoom)
                    if use_real_center:
                        pc = pc + self.get_pos_xyz(idx)
                self.pc_dict[idx] = pc
                self.norm_meta_dict[idx] = norm_meta
                self.pc_cache_idxs.append(idx)
                # check cache size
                if len(self.pc_cache_idxs) > self.cache_size:
                    pop_idx = self.pc_cache_idxs.popleft()
                    assert pop_idx in self.pc_dict, f'pop idx: {pop_idx} is not in pc cache'
                    del self.pc_dict[pop_idx]
        if idx not in self.pc_dict:
            if return_norm_meta:
                return None, None
            return None
        if return_norm_meta:
            return self.pc_dict[idx], self.norm_meta_dict[idx]
        return self.pc_dict[idx]

    def get_pcs(self, idxs, downsample=False, normalize=False, zoom=True, return_norm_meta=False, use_real_center=False):
        """ Return: list of N x 3 np array """
        pcs, norm_metas = [], []
        for idx in idxs:
            if return_norm_meta:
                pc, norm_meta = self.get_pc(idx, downsample, normalize, zoom, return_norm_meta, use_real_center)
                pcs.append(pc)
                norm_metas.append(norm_meta)
            else:
                pc = self.get_pc(idx, downsample, normalize, zoom, return_norm_meta, use_real_center)
                pcs.append(pc)
        if return_norm_meta:
            return pcs, norm_metas
        return pcs

    def get_fpfh(self, idx, radius_normal=0.05, radius_feature=0.05):
        """ Return: N x d np array """
        if idx not in self.fpfh_dict:
            # get pc
            pc = self.get_pc(idx)
            # create fpfh
            s = open3d.geometry.PointCloud()
            s.points = open3d.utility.Vector3dVector(pc)
            # normal
            s.estimate_normals(open3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
            # fpfh: np.ndarray
            self.fpfh_dict[idx] = open3d.pipelines.registration.compute_fpfh_feature(
                s, open3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
            self.fpfh_cache_idxs.append(idx)
            # check cache
            if len(self.fpfh_cache_idxs) > self.cache_size:
                pop_idx = self.fpfh_cache_idxs.popleft()
                assert pop_idx in self.fpfh_dict, f'pop idx: {pop_idx} is not in fpfh cache'
                del self.fpfh_dict[pop_idx]
        return self.fpfh_dict[idx]

    def get_fpfhs(self, idxs, radius_normal=0.05, radius_feature=0.05):
        """ Return: list of N x d np array """
        fpfhs = []
        for idx in idxs:
            fpfhs.append(self.get_fpfh(idx, radius_normal, radius_feature))
        return fpfhs

    def get_g_desc(self, pr_backbone, idx):
        """ Return: 1 x d np array """
        if idx not in self.g_desc_dict:
            # load global desc
            g_desc_pkl = os.path.join(self.g_desc_dir(pr_backbone), "{}.pickle".format(idx))
            if os.path.exists(g_desc_pkl):
                with open(g_desc_pkl, 'rb') as handle:
                    self.g_desc_dict[idx] = pickle.load(handle).reshape(1, -1)  # 1 x d
                    self.g_cache_idxs.append(idx)
            # check cache size
            if len(self.g_cache_idxs) > self.cache_size:
                pop_idx = self.g_cache_idxs.popleft()
                assert pop_idx in self.g_desc_dict, f'pop idx: {pop_idx} is not in global desc cache'
                del self.g_desc_dict[pop_idx]
        if idx not in self.g_desc_dict:
            return None
        return self.g_desc_dict[idx]

    def get_g_descs(self, pr_backbone, idxs):
        """ Return: list of 1 x d np array """
        g_descs = []
        for idx in idxs:
            g_descs.append(self.get_g_desc(pr_backbone, idx))
        return g_descs

    def get_l_kpt_desc(self, pr_backbone, idx, unify_coord=False, g_offset=True):
        """ Return: kpt: K x 3 np array; desc: K x d np array """
        if idx not in self.l_kpt_dict or idx not in self.l_desc_dict:
            # load local key points and desc
            l_desc_pkl = os.path.join(self.l_desc_dir(pr_backbone), "{}.pickle".format(idx))
            if os.path.exists(l_desc_pkl):
                with open(l_desc_pkl, 'rb') as handle:
                    l_kpt, l_desc, norm_meta = pickle.load(handle)
                    l_kpt = np.array(l_kpt, dtype=np.float64)
                    K = l_kpt.shape[0]
                    self.l_kpt_dict[idx], self.l_desc_dict[idx] = l_kpt.reshape(K, -1), l_desc.reshape(K, -1)
                    if norm_meta is not None:
                        self.norm_meta_dict[idx] = norm_meta
                    self.l_cache_idxs.append(idx)
            # check cache size
            if len(self.l_cache_idxs) > self.cache_size:
                pop_idx = self.l_cache_idxs.popleft()
                assert pop_idx in self.l_kpt_dict and pop_idx in self.l_desc_dict, f'pop idx: {pop_idx} is not in local kpt/desc cache'
                del self.l_kpt_dict[pop_idx]
                del self.l_desc_dict[pop_idx]
            if unify_coord and idx in self.norm_meta_dict and idx in self.l_kpt_dict:
                scale = self.norm_meta_dict[idx]['scale']  # 1 x 1
                trans = self.norm_meta_dict[idx]['trans'].reshape(1, self.norm_meta_dict[idx]['trans'].shape[0])  # 1 x 3
                if g_offset:
                    trans = trans + self.data_cfg['global_offset']
                self.l_kpt_dict[idx] = self.l_kpt_dict[idx] * scale + trans
        if idx not in self.l_kpt_dict or idx not in self.l_desc_dict:
            return None, None
        return self.l_kpt_dict[idx], self.l_desc_dict[idx]

    def get_l_kpts_descs(self, pr_backbone, idxs, unify_coord=False):
        """ Return: kpts: list of K x 3 np array; descs: list of K x d np array """
        l_kpts, l_descs = [], []
        for idx in idxs:
            l_kpt, l_desc = self.get_l_kpt_desc(pr_backbone, idx, unify_coord)
            l_kpts.append(l_kpt)
            l_descs.append(l_desc)
        return l_kpts, l_descs

    def get_knn_idxs(self, idx, k):
        """ Find kNN sub-maps in euclidean space
            1. use sub-maps in all ref trips if current_ref_trip_idx == -1.
            For training: set current_ref_trip_idx == -1
            For testing: set current_ref_trip_idx != -1
            Return: list
        """
        # find knn indices

        if self.current_ref_trip_idx == -1:
            sample_indices = list(range(len(self.records)))
        else:
            sample_indices = self.get_indices_in_dataset()
            sample_indices = sample_indices[self.current_ref_trip_idx]

        pos = self.get_pos_xy(idx).reshape(1, -1)
        if not self.data_cfg['is_test_dataset']:  # for training
            real_k = min(k * 2, self.positions.shape[0])
            _, index = self.position_tree.query(pos, k=real_k)
            index = np.random.choice(list(index[0]), k, replace=False)
        else:
            _, index = self.position_tree.query(pos, k=k)
            index = list(index[0])
        knn_idxs = np.array(sample_indices, dtype=int)[np.array(index, dtype=int)]
        return list(knn_idxs)

    def find_top_k_feat(self, model_type, top_k=300):
        """ top k in feature space """
        # skip empty data
        if len(self.records) == 0:
            return None

        # load global descriptors
        self.global_desc_list = self.get_g_descs(model_type, list(range(len(self.records))))

        # find top k
        n_q, n_p, n_n, n_u, n_valid = 0, 0, 0, 0, 0
        desc_dir = self.desc_dir(model_type)
        data_type = 'train' if self.for_training() else 'test'
        sample_indices = self.get_indices_in_dataset()
        for ref_trip_idx in tqdm(range(len(self.trip_names)), desc="Find and Save top k index"):
            database_indices = sample_indices[ref_trip_idx]
            database_descs = []
            for idx in database_indices:
                database_descs.append(self.dataset.global_desc_list[idx])
            database_descs = np.concatenate(database_descs, axis=0)
            database_tree = KDTree(database_descs)
            for query_trip_idx in range(len(self.trip_names)):
                if query_trip_idx == ref_trip_idx:
                    continue
                if not self.for_training():
                    if self.data_cfg['test_query_trips'] is not None and (
                            self.trip_names[query_trip_idx] not in self.data_cfg['test_query_trips']):
                        continue
                top_k_dict = dict()
                query_indices_in_dataset = sample_indices[query_trip_idx]
                for i in tqdm(range(len(query_indices_in_dataset))):
                    query_idx_in_dataset = query_indices_in_dataset[i]
                    a_tuple = self.get_tuple(query_idx_in_dataset, ref_trip_idx, skip_trip_itself=True)
                    if not a_tuple.positive_indices:  # no true pos means: it may be in test region!
                        continue
                    # search the nearest top k neighbors
                    if query_idx_in_dataset not in top_k_dict:
                        top_k_dict[query_idx_in_dataset] = {'top_k': [], 'state': []}
                    real_top_k = min(top_k, len(database_indices))
                    query_vec = self.global_desc_list[query_idx_in_dataset]
                    if query_vec is None:
                        continue
                    _, indices = database_tree.query(query_vec, k=real_top_k)
                    for j in indices[0]:
                        search_idx_in_dataset = database_indices[j]
                        top_k_dict[query_idx_in_dataset]['top_k'].append(search_idx_in_dataset)
                        if search_idx_in_dataset in a_tuple.positive_indices:  # positive
                            top_k_dict[query_idx_in_dataset]['state'].append(1)
                            n_p += 1
                        elif search_idx_in_dataset in a_tuple.negative_indices:  # negative
                            top_k_dict[query_idx_in_dataset]['state'].append(0)
                            n_n += 1
                        else:  # unknown (regard as negatives when testing)
                            top_k_dict[query_idx_in_dataset]['state'].append(0)
                            n_u += 1
                    n_q += 1
                top_k_pkl = os.path.join(desc_dir,
                                         f"top_k_index_{data_type}_{query_trip_idx}_{ref_trip_idx}_init.pickle")
                with open(top_k_pkl, 'wb') as handle:
                    pickle.dump(top_k_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print('save top k index in feature space: ', top_k_pkl)
        print(f'query: {n_q}, top k: {top_k}, pos ratio: {n_p / n_q / top_k}, '
              f'neg ratio: {n_n / n_q / top_k}, unknown ratio: {n_u / n_q / top_k}')
        print('n_valid: ', n_valid)

    def find_top_k_euc(self, top_k=300):
        if len(self.records) == 0:
            return
        top_k_dict = dict()
        euc_knn_dir = self.euc_knn_dir()
        check_makedirs(euc_knn_dir)
        basename_extra = 'test' if self.data_cfg['is_test_dataset'] else 'train'
        sample_indices = self.get_indices_in_dataset()
        for ref_trip_idx in tqdm(range(len(self.trip_names)), desc="Find and Save top k index"):
            indices_in_dataset = sample_indices[ref_trip_idx]
            self.set_current_ref_trip_idx(ref_trip_idx)
            self.construct_position_tree()
            top_k_dict = dict()
            real_k = min(top_k+1, len(indices_in_dataset))
            for i in tqdm(range(len(indices_in_dataset))):
                idx_in_dataset = indices_in_dataset[i]
                pos = self.get_pos_xy(idx_in_dataset).reshape(1,-1)
                _, index = self.position_tree.query(pos, k=real_k)
                index = list(index[0])[1:]  # skip itself
                top_k_dict[idx_in_dataset] = {
                    'euc_knn': np.array(indices_in_dataset, dtype=int)[np.array(index, dtype=int)]
                }
            top_k_pkl = os.path.join(euc_knn_dir,
                                     f"top_k_index_{basename_extra}_{ref_trip_idx}_init.pickle")
            with open(top_k_pkl, 'wb') as handle:
                pickle.dump(top_k_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('save top k index in feature space: ', top_k_pkl)

    def get_recall_precision(self, database_tree, database_indices, query_trip_idx, ref_trip_idx=-1, top_k=25,
                             skip_trip_itself=False):
        # get query indices in datasets
        sample_indices = self.get_indices_in_dataset()
        query_indices_in_dataset = sample_indices[query_trip_idx]
        # get recall
        num_evaluated = 0
        recall = np.zeros(top_k)
        precision = np.zeros(top_k)
        one_percent_retrieved = 0
        threshold = max(int(round(len(database_indices) / 100.0)), 1)
        real_top_k = top_k + 1
        if threshold + 1 > real_top_k:
            real_top_k = threshold + 1
        if ref_trip_idx == -1:
            if skip_trip_itself:
                ref_trip_name = "Not " + self.trip_names[query_trip_idx]
            else:
                ref_trip_name = "all trips"
        else:
            ref_trip_name = self.trip_names[ref_trip_idx]
        query_results = []
        add_one_more = (query_trip_idx == ref_trip_idx or ref_trip_idx == -1) and not skip_trip_itself
        progress_desc = "Get Recall@topN, {} & {}".format(self.trip_names[query_trip_idx], ref_trip_name)
        for i in tqdm(range(len(query_indices_in_dataset)), desc=progress_desc, leave=False):
            # tuple
            query_idx_in_dataset = query_indices_in_dataset[i]
            select_tuple = self.get_tuple(query_idx_in_dataset, ref_trip_idx, skip_trip_itself)
            true_positives = select_tuple.positive_indices
            if not true_positives:
                continue
            # search the nearest top k neighbors
            num_evaluated += 1
            query_centroid = self.records.loc[query_idx_in_dataset]
            query_vec = np.array([self.global_desc_list[query_idx_in_dataset]])
            if query_vec is None:
                continue
            search_indices_in_dataset = []
            _, indices = database_tree.query(query_vec, k=real_top_k)
            if add_one_more:  # the first is itself
                for j in range(1, len(indices[0])):
                    search_indices_in_dataset.append(database_indices[indices[0][j]])
            else:
                for j in range(len(indices[0])):
                    search_indices_in_dataset.append(database_indices[indices[0][j]])
            found_positive = False
            for j in range(len(search_indices_in_dataset)):
                if j >= top_k:
                    break
                search_idx_in_dataset = search_indices_in_dataset[j]
                if search_idx_in_dataset == query_idx_in_dataset:
                    continue
                if search_idx_in_dataset in true_positives:
                    if not found_positive:
                        recall[j] += 1
                        found_positive = True
                    precision[j] += 1

            # statistic recall
            query_result = {'query': query_centroid,
                            'state': 2,  # top1: 0, top1%: 1, fail: 2
                            'true_pos': self.records.loc[true_positives[0]],
                            'topN_files': [],
                            'topN_states': []}

            if len(list(set(search_indices_in_dataset[0:threshold]).intersection(set(true_positives)))) > 0:
                one_percent_retrieved += 1
                query_result['state'] = 1

            for j in range(5):
                query_result['topN_files'].append(self.records.loc[search_indices_in_dataset[j]])
                query_result['topN_states'].append(False)
                if search_indices_in_dataset[j] in true_positives:
                    query_result['topN_states'][j] = True
                    if j == 0:
                        query_result['state'] = 0
            query_results.append(query_result)
        # top k recall
        one_percent_recall = 0.0
        if num_evaluated > 0:
            one_percent_recall = (one_percent_retrieved / float(num_evaluated)) * 100
            recall = (np.cumsum(recall) / float(num_evaluated)) * 100
            # top k precision
            precision = (np.cumsum(precision) / float(num_evaluated)) * 100 / np.arange(1, top_k + 1, 1)
        return recall, precision, one_percent_recall, num_evaluated - one_percent_retrieved, threshold, query_results, num_evaluated, len(
            database_indices)

    @staticmethod
    def __get_hard_negatives(query_latent_vector, ref_latent_vectors, negative_indices, num_hard_neg=10):
        if len(negative_indices) < num_hard_neg:
            return []
        # build kdtree
        latent_vectors = ref_latent_vectors[negative_indices]
        latent_vec_tree = KDTree(latent_vectors)
        # select the nearest K negatives as the hardest samples
        _, indices = latent_vec_tree.query(np.array([query_latent_vector]), k=num_hard_neg)
        hard_negs = np.squeeze(np.array(negative_indices)[indices[0]])
        hard_negs = hard_negs.tolist()
        return hard_negs

    @staticmethod
    def __check_in_test_region(northing, easting, points, x_width, y_width):
        in_test_set = False
        for point in points:
            if point[0] - x_width < northing < point[0] + x_width and point[1] - y_width < easting < point[1] + y_width:
                in_test_set = True
                break
        return in_test_set

    def __get_training_tuple(self, query_idx_in_dataset, num_pos, num_neg, other_neg=False, normalize_cloud=True,
                             skip_trip_itself=False, load_overlap_indices=False):
        select_tuple = self.get_tuple(query_idx_in_dataset, -1, skip_trip_itself)
        norm_meta_data = []
        # query files
        origin_query_pcs = np.array([])
        query_files = [self.records.iloc[query_idx_in_dataset]['file']]
        origin_query_files = []
        for f in query_files:
            f_name = os.path.basename(f)
            f_dir = os.path.dirname(f)
            origin_f = os.path.join(f_dir, 'origin_bin', f_name)
            origin_query_files.append(origin_f)
        if self.data_cfg['self_collected']:
            query_pcs = load_pc_files(query_files, "", use_np_load=True)
            origin_query_pcs = load_pc_files(origin_query_files, "", use_np_load=True)
            origin_query_pcs = np.array(origin_query_pcs)
            if normalize_cloud:
                query_pcs, tmp_norm_meta = normalize_point_clouds(query_pcs, True)
                norm_meta_data += tmp_norm_meta
        else:
            query_pcs = load_pc_files(query_files, "")
        query_pcs = np.array(query_pcs)
        # positives
        positive_indices = []
        origin_positive_pcs = np.array([])
        positive_pcs = np.array([])
        if len(select_tuple.positive_indices) >= num_pos:
            positive_indices = random.sample(select_tuple.positive_indices, num_pos)
            positive_files = self.records.loc[positive_indices]['file'].values.tolist()
            origin_positive_files = []
            for f in positive_files:
                f_name = os.path.basename(f)
                f_dir = os.path.dirname(f)
                origin_f = os.path.join(f_dir, 'origin_bin', f_name)
                origin_positive_files.append(origin_f)
            if self.data_cfg['self_collected']:
                positive_pcs = load_pc_files(positive_files, "", use_np_load=True)
                origin_positive_pcs = load_pc_files(origin_positive_files, "", use_np_load=True)
                origin_positive_pcs = np.array(origin_positive_pcs)
                if normalize_cloud:
                    positive_pcs, tmp_norm_meta = normalize_point_clouds(positive_pcs, True)
                    norm_meta_data += tmp_norm_meta
            else:
                positive_pcs = load_pc_files(positive_files, "")
            positive_pcs = np.array(positive_pcs)
        # negatives
        negative_indices = []
        origin_negative_pcs = np.array([])
        negative_pcs = np.array([])
        if len(select_tuple.negative_indices) >= num_neg:
            hard_negative_indices = self.get_hard_negative_indices(query_idx_in_dataset)
            if len(hard_negative_indices) >= num_neg:
                negative_indices = random.sample(hard_negative_indices, num_neg)
            else:
                negative_indices = hard_negative_indices
            while len(negative_indices) < num_neg:
                neg_idx = random.choice(select_tuple.negative_indices)
                if neg_idx not in negative_indices:
                    negative_indices.append(neg_idx)
            negative_files = self.records.loc[negative_indices]['file'].values.tolist()
            origin_negative_files = []
            for f in negative_files:
                f_name = os.path.basename(f)
                f_dir = os.path.dirname(f)
                origin_f = os.path.join(f_dir, 'origin_bin', f_name)
                origin_negative_files.append(origin_f)
            if self.data_cfg['self_collected']:
                negative_pcs = load_pc_files(negative_files, "", use_np_load=True)
                origin_negative_pcs = load_pc_files(origin_negative_files, "", use_np_load=True)
                origin_negative_pcs = np.array(origin_negative_pcs)
                if normalize_cloud:
                    negative_pcs, tmp_norm_meta = normalize_point_clouds(negative_pcs, True)
                    norm_meta_data += tmp_norm_meta
            else:
                negative_pcs = load_pc_files(negative_files, "")
            negative_pcs = np.array(negative_pcs)
        # other negatives for Quadruplet loss
        other_neg_indices = []
        origin_other_neg_pcs = np.array([])
        other_neg_pcs = np.array([])
        if other_neg:
            neighbors = select_tuple.positive_indices
            for neg_idx_in_dataset in negative_indices:
                select_tuple2 = self.get_tuple(neg_idx_in_dataset, -1, skip_trip_itself)
                neighbors = neighbors + select_tuple2.positive_indices
            if skip_trip_itself:
                sum_record_size = 0
                all_indices = []
                for i in range(len(self.trip_names)):
                    record_size = self.records_size_list[i]
                    indices_in_trip = np.arange(sum_record_size, sum_record_size + record_size).tolist()
                    sum_record_size += record_size
                    all_indices += indices_in_trip
            else:
                all_indices = self.records.index.values.tolist()
            possible_neg_indices = list(set(all_indices) - set(neighbors))
            if len(possible_neg_indices) > 0:
                possible_neg_indices = np.random.choice(possible_neg_indices)
                other_neg_indices = [possible_neg_indices]
                other_neg_files = [self.records.loc[possible_neg_indices]['file']]
                origin_other_neg_files = []
                for f in other_neg_files:
                    f_name = os.path.basename(f)
                    f_dir = os.path.dirname(f)
                    origin_f = os.path.join(f_dir, 'origin_bin', f_name)
                    origin_other_neg_files.append(origin_f)
                if self.data_cfg['self_collected']:
                    other_neg_pcs = load_pc_files(other_neg_files, "", use_np_load=True)
                    origin_other_neg_pcs = load_pc_files(origin_other_neg_files, "", use_np_load=True)
                    origin_other_neg_pcs = np.array(origin_other_neg_pcs)
                    if normalize_cloud:
                        other_neg_pcs, tmp_norm_meta = normalize_point_clouds(other_neg_pcs, True)
                        norm_meta_data += tmp_norm_meta
                else:
                    other_neg_pcs = load_pc_files(other_neg_files, "")
                other_neg_pcs = np.array(other_neg_pcs)
        res = {'indices': ([query_idx_in_dataset], positive_indices, negative_indices, other_neg_indices),
               'input_cloud': (query_pcs, positive_pcs, negative_pcs, other_neg_pcs),
               'input_norm': norm_meta_data,
               'origin_cloud': (origin_query_pcs, origin_positive_pcs, origin_negative_pcs, origin_other_neg_pcs)}
        if load_overlap_indices:
            res['overlap_indices'] = self.get_overlap_indices(query_idx_in_dataset, positive_indices)
        return res


if __name__ == '__main__':
    # # top k in euclidean space
    # for key in dataset_info_dict:
    #     # train data
    #     dataset_i = SceneDataSet(key, True)
    #     dataset_i.load(query_trip_indices=-1, skip_trip_itself=False)
    #     dataset_i.find_top_k_euc()
    #     # test data
    #     dataset_i = SceneDataSet(key, False)
    #     dataset_i.load(query_trip_indices=-1, skip_trip_itself=True)
    #     dataset_i.find_top_k_euc()

    # print dataset info
    # for key in dataset_info_dict:
    #     # train data
    #     dataset_i = SceneDataSet(key, True)
    #     dataset_i.print_stat_info()
    #     # test data
    #     dataset_i = SceneDataSet(key, False)
    #     dataset_i.print_stat_info()
    #     print()

    # create datasets from pb file
    datasets = ['hankou_train']
    for dataset_name in datasets:
        # train data
        dataset_from_pb = SceneDataSet(dataset_name, True)
        dataset_from_pb.create_from_pb(max_neg=15000)
        # test data
        dataset_from_pb = SceneDataSet(dataset_name, False)
        dataset_from_pb.create_from_pb(max_neg=15000)
