from torch.utils import data
from sklearn.neighbors import KDTree

from utils.loading_pointclouds import *
from datasets.scene_dataset import *
from datasets.dataset_info import *


class PlaceRecognitionDataSet(data.Dataset):
    def __init__(self, name, for_training, num_pos=2, num_neg=14, other_neg=True, data_augmentation=None,
                 num_hard_neg=10, num_sample_neg=3000, normalize_cloud=True,
                 load_overlap_indices=False):
        self.dataset = SceneDataSet(name, for_training)
        self.dataset.load(query_trip_indices=-1, skip_trip_itself=self.dataset.data_cfg['skip_trip_itself'])
        print("load datasets: {}, trips: {}".format(self.dataset.data_dir(), self.dataset.trip_names))
        self.num_pos = num_pos
        self.num_neg = num_neg
        self.other_neg = other_neg
        self.data_augmentation = data_augmentation  # 'rotate' / 'jitter' / 'random'
        self.num_hard_neg = num_hard_neg
        self.num_sample_neg = num_sample_neg
        self.normalize_cloud = normalize_cloud
        if self.data_augmentation is None:
            self.data_augmentation = []
        self.load_overlap_indices = load_overlap_indices

    def __getitem__(self, index):
        # load tuples within a query and reference trips in one scene
        query_trip_idx, _ = self.dataset.get_query_idx_in_trip(index)
        # augment data
        data_tuple = self.__augment_tuple(index)
        return data_tuple

    def __len__(self):
        return len(self.dataset.records)

    def update_global_descs(self, model, cfg, device, batch_size=18, stat_time=False, save=False):
        print("update global descriptors: {}, trips: {}".format(self.dataset.data_dir(), self.dataset.trip_names))
        self.dataset.make_descs(model, cfg, device, batch_size=batch_size, stat_time=stat_time, save=save)

    def clear_global_descs(self):
        print("clear global descriptors: {}, trips: {}".format(self.dataset.data_dir(), self.dataset.trip_names))
        self.dataset.clear_global_descs()

    def find_and_save_top(self, model_type=None, top_k=300, space_type='feature'):
        print(f"Find and Save top k info in {space_type} space")
        if space_type == 'feature' or space_type == 'feat':
            self.dataset.find_top_k_feat(model_type, top_k)
        elif space_type == 'euclidean' or space_type == 'euc':
            self.dataset.find_top_k_euc(top_k)

    def get_recall_precision(self, top_k=25):
        recall_dict = dict()
        print("get recall: {}, trips: {}".format(self.dataset.data_dir(), self.dataset.trip_names))
        # get indices in datasets
        sample_indices = self.dataset.get_indices_in_dataset()
        # build kd tree and get recall one by one
        for ref_trip_idx in range(len(self.dataset.trip_names)):
            database_indices = sample_indices[ref_trip_idx]
            database_descs = []
            for idx in database_indices:
                database_descs.append(self.dataset.global_desc_list[idx])
            database_descs = np.stack(database_descs, axis=0)
            database_tree = KDTree(database_descs)
            for query_trip_idx in range(len(self.dataset.trip_names)):
                if self.dataset.data_cfg['skip_trip_itself'] and query_trip_idx == ref_trip_idx:
                    continue
                if self.dataset.data_cfg['test_query_trips'] is not None and (self.dataset.trip_names[query_trip_idx] not in self.dataset.data_cfg['test_query_trips']):
                    continue
                self.dataset.load([query_trip_idx], self.dataset.data_cfg['skip_trip_itself'])
                recall_dict[query_trip_idx, ref_trip_idx] = self.dataset.get_recall_precision(
                    database_tree, database_indices, query_trip_idx, ref_trip_idx, top_k=top_k,
                    skip_trip_itself=self.dataset.data_cfg['skip_trip_itself'])
        return recall_dict

    def get_trip_name(self, trip_idx):
        return self.dataset.trip_names[trip_idx]

    def __augment_tuple(self, query_idx_in_dataset):
        data_tuple = self.dataset.get_query_pos_neg_tuple(
            query_idx_in_dataset, self.num_pos, self.num_neg,
            self.other_neg, self.num_hard_neg, self.num_sample_neg, self.normalize_cloud,
            self.dataset.data_cfg['skip_trip_itself'], self.load_overlap_indices)
        if 'random' in self.data_augmentation:
            rand_num = np.random.randint(low=0, high=2)
            if rand_num == 0:
                self.__augment_pcs(data_tuple, 'rotate')
            else:
                self.__augment_pcs(data_tuple, 'jitter')
        else:
            if 'rotate' in self.data_augmentation:
                self.__augment_pcs(data_tuple, 'rotate')
            if 'jitter' in self.data_augmentation:
                self.__augment_pcs(data_tuple, 'jitter')
        return data_tuple

    @staticmethod
    def __augment_pcs(data_tuple, aug_type='rotate'):
        query_pcs, pos_pcs, neg_pcs, neg2_pcs = data_tuple['input_cloud']
        norm_meta_data = data_tuple['input_norm']
        origin_query_pcs, origin_pos_pcs, origin_neg_pcs, origin_neg2_pcs = data_tuple['origin_cloud']
        aug_query_pcs, aug_pos_pcs, aug_neg_pcs, aug_neg2_pcs = None, None, None, None
        query_norm_meta = norm_meta_data[:query_pcs.shape[0]]
        pos_norm_meta = norm_meta_data[query_pcs.shape[0]:query_pcs.shape[0] + pos_pcs.shape[0]]
        neg_norm_meta = norm_meta_data[query_pcs.shape[0] + pos_pcs.shape[0]:query_pcs.shape[0] + pos_pcs.shape[0] + neg_pcs.shape[0]]
        neg2_norm_meta = norm_meta_data[query_pcs.shape[0] + pos_pcs.shape[0] + neg_pcs.shape[0]:len(norm_meta_data)]
        if aug_type == 'rotate':
            aug_query_pcs, aug_query_norm_meta = rotate_point_cloud(query_pcs, query_norm_meta)
            aug_pos_pcs, aug_pos_norm_meta = rotate_point_cloud(pos_pcs, pos_norm_meta)
            aug_neg_pcs, aug_neg_norm_meta = rotate_point_cloud(neg_pcs, neg_norm_meta)
            aug_neg2_pcs, aug_neg2_norm_meta = rotate_point_cloud(neg2_pcs, neg2_norm_meta)
        elif aug_type == 'jitter':
            aug_query_pcs = jitter_point_cloud(query_pcs)
            aug_pos_pcs = jitter_point_cloud(pos_pcs)
            aug_neg_pcs = jitter_point_cloud(neg_pcs)
            aug_neg2_pcs = jitter_point_cloud(neg2_pcs)
            aug_query_norm_meta = query_norm_meta
            aug_pos_norm_meta = pos_norm_meta
            aug_neg_norm_meta = neg_norm_meta
            aug_neg2_norm_meta = neg2_norm_meta
        query_pcs = np.vstack((query_pcs, aug_query_pcs))
        pos_pcs = np.vstack((pos_pcs, aug_pos_pcs))
        neg_pcs = np.vstack((neg_pcs, aug_neg_pcs))
        neg2_pcs = np.vstack((neg2_pcs, aug_neg2_pcs))
        query_norm_meta += aug_query_norm_meta
        pos_norm_meta += aug_pos_norm_meta
        neg_norm_meta += aug_neg_norm_meta
        neg2_norm_meta += aug_neg2_norm_meta
        aug_norm_meta = []
        aug_norm_meta += query_norm_meta
        aug_norm_meta += pos_norm_meta
        aug_norm_meta += neg_norm_meta
        aug_norm_meta += neg2_norm_meta
        norm_meta_data = aug_norm_meta
        origin_query_pcs = np.vstack((origin_query_pcs, origin_query_pcs))
        origin_pos_pcs = np.vstack((origin_pos_pcs, origin_pos_pcs))
        origin_neg_pcs = np.vstack((origin_neg_pcs, origin_neg_pcs))
        origin_neg2_pcs = np.vstack((origin_neg2_pcs, origin_neg2_pcs))
        data_tuple['input_cloud'] = query_pcs, pos_pcs, neg_pcs, neg2_pcs
        data_tuple['input_norm'] = norm_meta_data
        data_tuple['origin_cloud'] = origin_query_pcs, origin_pos_pcs, origin_neg_pcs, origin_neg2_pcs


def create_dataset_batch(name, for_training):
    cfg = dataset_info_dict[name].train_cfg() if for_training else dataset_info_dict[name].test_cfg()
    dataset = SceneDataSet(name, for_training)
    dataset.create(cloud_ext=cfg['cloud_ext'], trip_names=cfg['trip_names'],
                   test_region_vertices=cfg['test_region_vertices'],
                   test_region_width=cfg['test_region_width'],
                   search_radius_pos=cfg['search_radius_pos'],
                   search_radius_neg=cfg['search_radius_neg'])


def create_pointnet_vlad_dataset():
    # create training / test DataSet
    print("create Place Recognition DataSet")
    dataset_list = [
        # 'oxford',
        # 'university',
        # 'residential',
        # 'business',
        # 'sejong',
        # 'dcc_20m',
        # 'kitti360_20m'
        # 'dcc_5m',
        # 'kitti360_5m',
        # 'hankou_train',
        'cs_college',
        # 'info_campus',
        # 'zhongshan_park',
        # 'jiefang_road',
        # 'yanjiang_road1',
        # 'yanjiang_road2'
    ]
    for dataset_name in dataset_list:
        print('Create Place Recognition Dataset: ', dataset_name)
        create_dataset_batch(dataset_name, for_training=True)
        create_dataset_batch(dataset_name, for_training=False)


if __name__ == '__main__':
    create_pointnet_vlad_dataset()
