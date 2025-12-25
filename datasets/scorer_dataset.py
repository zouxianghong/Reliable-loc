import os
import pickle
import random

import numpy as np
from tqdm import tqdm
import torch
from torch.utils import data

from utils.util import check_makedirs
from utils.train_util import get_device
from datasets.scene_dataset import mycollate
from datasets.pose_est_dataset import PoseEstDataset, GroupType
from pose_estimation.pose_est_ransac import compute_RRE_RTE, PoseEstimator
from pose_estimation.pose_est_net import PoseEstMatcher

# make one hot code for scorer
def make_one_hot_code_i(value, bins=None):
    """ bins: [b0, b1), [b1, b2), ..., [bn-2, bn-1), [bn-1, +inf) """
    if bins is None:
        bins = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    one_hot_code = np.zeros(len(bins))
    if value <= bins[0]:
        cls = 0
        one_hot_code[0] = 1
    elif value >= bins[-1]:
        cls = len(bins) - 1
        one_hot_code[-1] = 1
    for i in range(1, len(bins)):
        if value < bins[i]:
            cls = i-1
            one_hot_code[i-1] = 1
            break
    return one_hot_code, cls


# make one hot code for scorer
def make_one_hot_code(R_value, t_value, R_bins=None, t_bins=None):
    one_hot_code = np.zeros(len(R_bins))
    _, R_cls = make_one_hot_code_i(R_value, R_bins)
    _, t_cls = make_one_hot_code_i(t_value, t_bins)
    cls = R_cls if R_cls > t_cls else t_cls
    one_hot_code[cls] = 1
    return one_hot_code, cls


# get predict class and confidence
def get_score_cls_conf(pred_scores):
    highest_cls, highest_conf = -1, 0.0
    for i in range(len(pred_scores)):
        if pred_scores[i] > highest_conf:
            highest_conf = pred_scores[i]
            highest_cls = i
    return highest_cls, highest_conf


class ScorerDataset(data.Dataset):
    """ Dataset for pose estimation scorer """
    def __init__(self, data_cfg):
        super(ScorerDataset, self).__init__()
        self.data_dir = data_cfg['data_dir']
        self.is_test_dataset = data_cfg['is_test_dataset']
        dataset_type_str = 'test' if self.is_test_dataset else 'train'
        summary_pkl = os.path.join(self.pickle_dir(), f'scorer_{dataset_type_str}', 'summary.pickle')
        self.select_idxs = []
        self.R_bins, self.t_bins = [], []
        if os.path.exists(summary_pkl):
            with open(summary_pkl, 'rb') as handle:
                self.select_idxs, self.R_bins, self.t_bins = pickle.load(handle)
                print('load summary_pkl for scorer: {}'.format(summary_pkl))
    
    def pickle_dir(self, pickle_dir_name='pickle_data'):
        return os.path.join(self.data_dir, pickle_dir_name)

    def __getitem__(self, index):
        idx = self.select_idxs[index]
        dataset_type_str = 'test' if self.is_test_dataset else 'train'
        item_pkl = os.path.join(self.pickle_dir(), f'scorer_{dataset_type_str}', f'{idx}.pickle')
        if os.path.exists(item_pkl):
            with open(item_pkl, 'rb') as handle:
                a_kpt, a_feat, k_kpt, k_feat, T_est, Rt_cls = pickle.load(handle)
                Rt_one_hot = np.zeros(len(self.R_bins))
                Rt_one_hot[Rt_cls] = 1
                return {
                    'a_kpt': a_kpt,
                    'a_feat': a_feat,
                    'k_kpt': k_kpt,
                    'k_feat': k_feat,
                    'T_est': T_est,
                    'Rt_cls': Rt_cls,
                    'Rt_one_hot': Rt_one_hot
                }
        return None

    def __len__(self):
        return len(self.select_idxs)

    def parse_batch_data(self, in_batch_data, device):
        # k_kpt = T_est * a_kpt
        bdata = { 'a_kpt': [], 'a_feat': [], 'k_kpt': [], 'k_feat': [],
                  'Rt_cls': [], 'Rt_one_hot': []}
        for item_i in in_batch_data:
            if item_i is None:
                continue
            a_kpt_i, a_feat_i = item_i['a_kpt'], item_i['a_feat']
            k_kpt_i, k_feat_i = item_i['k_kpt'], item_i['k_feat']
            T_est_i, Rt_one_hot_i = item_i['T_est'], item_i['Rt_one_hot']
            Rt_cls_i = item_i['Rt_cls']
            a_kpt_i = a_kpt_i @ T_est_i[:3, :3].T + T_est_i[:3, 3:].T  # m x 3
            bdata['a_kpt'].append(a_kpt_i[None, ...])  # 1 x m x 3
            bdata['a_feat'].append(a_feat_i[None, ...])  # 1 x m x d
            bdata['k_kpt'].append(k_kpt_i[None, ...])  # 1 x m x 3
            bdata['k_feat'].append(k_feat_i[None, ...])  # 1 x m x d
            bdata['Rt_cls'].append(np.array([[Rt_cls_i]]))  # 1 x 1
            bdata['Rt_one_hot'].append(Rt_one_hot_i[None, ...])  # 1 x num_Rt_class
        if len(bdata['a_kpt']) == 0:
            return None
        # cat / convert to tensor
        for key in bdata:
            bdata[key] = np.concatenate(bdata[key], axis=0)
            bdata[key] = torch.from_numpy(bdata[key]).to(device)
        return bdata

    @staticmethod
    def check_Rt_dict(selected_Rt_dict, Rt_count):
        prior_Rt_key = list(selected_Rt_dict.keys())[0]
        prior_Rt_count = len(selected_Rt_dict[prior_Rt_key])
        for key in selected_Rt_dict:
            if len(selected_Rt_dict[key]) < prior_Rt_count:
                prior_Rt_key = key
                prior_Rt_count = len(selected_Rt_dict[key])
        if prior_Rt_count < Rt_count:
            return prior_Rt_key, prior_Rt_count
        else:
            return None, None

    def prepare_data(self, matcher_model_file, group_type=GroupType.FOR_TRAINING, select_data_len=4000, max_delta_count=10):
        # load matcher model and config
        checkpoint = torch.load(matcher_model_file)
        cfg = checkpoint['config']
        matcher = PoseEstMatcher(cfg)
        matcher.load_state_dict(checkpoint['state_dict_encoder'])
        # get device
        device = get_device(cfg)
        matcher.to(device)
        matcher.eval()
        # pose est datasets / loader
        t_dataset = PoseEstDataset(cfg['train_data_dir'], cfg['submap_type'], self.is_test_dataset,
                                   top_k=cfg['top_k'], group_type=group_type)
        #t_dataset.top_k_dict_keys = t_dataset.top_k_dict_keys[:select_data_len]
        t_loader = data.DataLoader(t_dataset, batch_size=16, shuffle=False, num_workers=4,
                                   pin_memory=True, collate_fn=mycollate)
        estimator = PoseEstimator(cfg)
        # save dir
        dataset_type_str = 'test' if self.is_test_dataset else 'train'
        save_dir = os.path.join(self.pickle_dir(), f'scorer_{dataset_type_str}')
        check_makedirs(save_dir)
        # run pose estimation and save result
        R_bins, t_bins = [0.0, 2.5, 5.0, 10.0, 20.0], [0.0, 1.0, 2.0, 4.0, 8.0]
        data_to_save, Rt_dict = [], dict()
        for i in range(len(R_bins)):
            Rt_dict[i] = []
        for data_item in tqdm(t_loader, desc="Evaluate One Epoch-{}".format(dataset_type_str)):
            # batch data
            bdata = t_dataset.parse_batch_data(data_item, device, num_gp_train=4)
            if bdata is None:
                continue
            b, m, d = bdata['a_feat'].size()
            a_kpt, a_feat = bdata['a_kpt'].contiguous(), bdata['a_feat'].contiguous()
            k_kpt, k_feat = bdata['k_kpt'].contiguous(), bdata['k_feat'].contiguous()
            k_extra_score = bdata['k_score'].contiguous() if cfg['use_rerank_sim'] else None
            T_gt = bdata['T_gt']
            with torch.no_grad():
                pred = matcher(a_kpt, a_feat, k_kpt, k_feat, k_extra_score)
            # estimate pose
            b_idxs, T_est, _, _ = estimator.run(a_kpt, k_kpt, pred['matches0'], pred['matching_scores0'])
            a_kpt, a_feat = a_kpt.detach().cpu().numpy(), pred['source_final'].detach().cpu().numpy()
            k_kpt, k_feat = k_kpt.detach().cpu().numpy(), pred['target_final'].detach().cpu().numpy()
            for i in range(b):
                if i not in b_idxs:
                    continue
                idx = b_idxs.index(i)
                RRE, RTE = compute_RRE_RTE(T_est[idx], T_gt[i])
                RTE = RTE * t_dataset.scale
                a_kpt_i, a_feat_i, k_kpt_i, k_feat_i = a_kpt[i], a_feat[i], k_kpt[i], k_feat[i]
                one_hot, cls = make_one_hot_code(RRE, RTE, R_bins, t_bins)
                data_to_save.append(cls)
                Rt_dict[cls].append(len(data_to_save))
                item_pkl = os.path.join(save_dir, f'{len(data_to_save)-1}.pickle')
                with open(item_pkl, 'wb') as handle:
                    pickle.dump((a_kpt_i, a_feat_i, k_kpt_i, k_feat_i, T_est[idx], cls), handle,
                                protocol=pickle.HIGHEST_PROTOCOL)
            # min_count = 1e10
            # for key in Rt_dict:
            #     if len(Rt_dict[key]) < min_count:
            #         min_count = len(Rt_dict[key])
            # if min_count > 25:
            #     break
        # select data by cls, make cls balance
        Rt_count = 1e10
        for key in Rt_dict:
            if len(Rt_dict[key]) < Rt_count:
                Rt_count = len(Rt_dict[key])
        selected_idxs, selected_Rt_dict = [], dict()
        for key in Rt_dict:
            selected_Rt_dict[key] = []
        max_Rt_count, max_Rt_cls = -1, -1
        while True:
            prior_Rt_key, prior_Rt_count = self.check_Rt_dict(selected_Rt_dict, Rt_count)
            if prior_Rt_key is None:
                break
            found = False
            tmp_idxs = list(range(len(data_to_save)))
            random.shuffle(tmp_idxs)
            for i in tmp_idxs:
                if i in selected_idxs:
                    continue
                Rt_cls = data_to_save[i]
                if Rt_cls == prior_Rt_key:
                    if Rt_cls == max_Rt_cls and max_Rt_count - prior_Rt_count > max_delta_count:
                        continue
                    found = True
                    selected_idxs.append(i)
                    selected_Rt_dict[Rt_cls].append(i)
                    if len(selected_Rt_dict[Rt_cls]) > max_Rt_count:
                        max_Rt_count = len(selected_Rt_dict[Rt_cls])
                        max_Rt_cls = Rt_cls
                    break
            if not found:
                break
        # save selected data
        summary_pkl = os.path.join(save_dir, 'summary.pickle')
        with open(summary_pkl, 'wb') as handle:
            pickle.dump((selected_idxs, R_bins, t_bins), handle, protocol=pickle.HIGHEST_PROTOCOL)
        # remove useless files
        n_remove = 0
        for i in range(len(data_to_save)):
            if i not in selected_idxs:
                item_pkl = os.path.join(save_dir, f'{i}.pickle')
                os.remove(item_pkl)
                n_remove = n_remove + 1
        print('remove {} unused files'.format(n_remove))


if __name__ == '__main__':
    # prepare datav
    data_dirs = ["/home/ericxhzou/Code/benchmark_datasets/wh_hankou_origin",
                 #"/home/ericxhzou/Code/benchmark_datasets/whu_campus_origin"
                 ]
    matcher_model_file = '/home/ericxhzou/Code/ppt-net-plus/weights/matcher.pth'
    for data_dir in data_dirs:
        # for training
        scorer_dataset = ScorerDataset(data_dir=data_dir, is_test_dataset=False)
        scorer_dataset.prepare_data(matcher_model_file=matcher_model_file, group_type=GroupType.FOR_TRAINING, select_data_len=4000)
        # for testing
        scorer_dataset.is_test_dataset = True
        scorer_dataset.prepare_data(matcher_model_file=matcher_model_file, group_type=GroupType.GROUP_TOP_K, select_data_len=1000)