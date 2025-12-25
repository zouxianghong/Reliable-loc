from enum import Enum

import torch.nn.functional
from torch.utils import data

import utils.util
from libs.pointops.functions import pointops
from datasets.scene_dataset import *
from utils.util import *
from utils.train_util import nn_dist_np


# divide top k into groups by absolute distance
def divide_topk_into_groups(k_pos, dist_thresh=30.0):
    groups = [[0]]
    dists = nn_dist_np(k_pos)  # k x k
    for i in range(1, len(k_pos)):
        found = False
        for j in range(len(groups)):
            if found:
                break
            found = True
            for k in groups[j]:
                if dists[i, k] > dist_thresh:
                    found = False
                    break
            if found:
                groups[j].append(i)
        if not found:
            groups.append([i])
    return groups


# merge top k clouds
def merge_topk(k_kpts, scale=30.0):
    """ k_kpts: k x n x 3 or n x 3
        out: k x n x 3 or n x 3
    """
    if len(k_kpts.shape) == 3:
        k_center = np.mean(np.mean(k_kpts, axis=0), axis=0, keepdims=True)  # 1 x 3
    else:
        k_center = np.mean(k_kpts, axis=0, keepdims=True)  # 1 x 3
    k_kpts = (k_kpts - k_center) / scale
    return k_kpts, k_center


class GroupType(Enum):
    """ How to deal with top k """
    FOR_TRAINING = 0  # include all kinds of groups, for training
    ONLY_TOP_1 = 1  # only use top 1, refer to DH3D, LCDNet, EgoNN, PADLoC
    INDIVIDUAL_TOP_K = 2  # estimate transformation between query and top k one by one, refer to Rank-PointRetrieval, SpectralGV
    WHOLE_TOP_K = 3  # regard top k as a whole entity
    GROUP_TOP_K = 4  # estimate transformation between query and grouped top k

    def get_str(self):
        if self.name == GroupType.FOR_TRAINING.name:
            return 'for_training'
        elif self.name == GroupType.ONLY_TOP_1.name:
            return 'only_top_1'
        elif self.name == GroupType.INDIVIDUAL_TOP_K.name:
            return 'individual_top_k'
        elif self.name == GroupType.WHOLE_TOP_K.name:
            return 'whole_top_k'
        else:
            return 'group_top_k'


class SamplingType(Enum):
    FPS = 0  # Furthest point sampling
    RANDOM = 1  # Random point sampling
    CORRELATION = 2  # Correlation point sampling

    def str(self):
        if self.name == SamplingType.FPS.name:
            return 'fps'
        elif self.name == SamplingType.RANDOM.name:
            return 'rps'
        elif self.name == SamplingType.CORRELATION.name:
            return 'cps'
        else:
            assert 'Invalid point sampling type!'


class PoseEstResult():
    def __init__(self):
        self.T_est = np.eye(4)
        self.overlap = -1.e10
        self.pps = None
        self.regression_score = 0.0


class EvaluationResult():
    def __init__(self):
        self.RRE = 1.e10
        self.RTE = 1.e10
        self.run_time = 0.0


class PoseEstDataset(data.Dataset):
    """ Dataset for pose estimation based on top k """
    def __init__(self, name, for_training, top_k=5, scale=30.0,
                 group_type=GroupType.ONLY_TOP_1, expand_group=False, sample_type=SamplingType.RANDOM):
        super(PoseEstDataset, self).__init__()
        self.dataset = SceneDataSet(name, for_training)
        self.dataset.load(query_trip_indices=None)
        self.pr_backbone = 'patch_aug_net'
        self.top_k_dict = None
        self.top_k_dict_keys = []
        self.top_k = top_k
        self.scale = scale
        self.group_type = group_type  # only top 1 / top k individual / top k group
        self.expand_group = expand_group
        self.sample_type = sample_type

    def __getitem__(self, index):
        # query and top k indices
        a_idx = self.top_k_dict_keys[index]
        # match dir
        dataset_type_str = 'test' if self.dataset.data_cfg['is_test_dataset'] else 'train'
        sample_type_str = self.sample_type.str()
        item_dir = os.path.join(self.dataset.pickle_dir(), f'pose_est_top{self.top_k}_{dataset_type_str}_{sample_type_str}')
        item_pkl = os.path.join(item_dir, "{}.pickle".format(a_idx))
        item = None
        if os.path.exists(item_pkl):
            with open(item_pkl, 'rb') as handle:
                item = pickle.load(handle)
        # 'gp_state': bool
        # 'a_idx': int
        # 'a_kpt': m x 3
        # 'a_feat': m x d
        # 'T_a': 4 x 4, T_a' = T_a * T_rnd.inv()
        # 'k_groups': k_groups -> 'k_idxs', 'k_states',  'k_pos',
        #                         'center_idx',  'center_knn_idx',   'T_k',     'T_gt',
        #                         'pairs', 'unpair0',   'unpair1', 'matches0'
        # 'k_idxs': list of int
        # 'k_states': list of bool
        # 'k_pos': k x 3
        # 'k_kpt': k x m x 3
        # 'k_feat': k x m x d
        return item

    def __len__(self):
        return len(self.top_k_dict_keys)

    @staticmethod
    def make_random_transformation():
        R_rnd = random_rotation_matrix()
        t_rnd = (np.random.randn(3, 1) - 0.5) * 2
        T_rnd = np.block([[R_rnd, t_rnd],
                         [np.zeros([1,3]), np.ones([1,1])]])
        return T_rnd

    def load_desc_by_idxs(self, idxs, unify_coord=True):
        # top k desc and pos
        global_descs, local_descs, local_poss, positions = [], [], [], []
        for idx in idxs:
            position = self.dataset.get_pos_xyz(idx)
            global_desc = self.dataset.get_g_desc(self.pr_backbone, idx)
            local_pos, local_desc = self.dataset.get_l_kpt_desc(self.pr_backbone, idx, unify_coord)
            local_desc = local_desc.reshape(1, local_desc.shape[0], local_desc.shape[1])  # 1 x k x d
            local_pos = local_pos.reshape(1, local_pos.shape[0], local_pos.shape[1])  # 1 x k x 3
            global_descs.append(global_desc)
            local_descs.append(local_desc)
            local_poss.append(local_pos)
            positions.append(position.reshape(1, -1))
        # cat
        g_desc = np.concatenate(global_descs, axis=0)
        l_desc = np.concatenate(local_descs, axis=0)
        l_pos = np.concatenate(local_poss, axis=0)
        pos = np.concatenate(positions, axis=0)
        # pos: k x 3, g_desc: k x d, l_pos: k x n x 3, l_desc: k x n x d
        return pos, g_desc, l_pos, l_desc

    def parse_group_data(self, item_i, device, num_gp_train=4):
        # anchor: idx, T, center, kpt, feat
        a_idx, T_rnd, T_a, a_center = item_i['a_idx'], item_i['T_rnd'], item_i['T_a'], item_i['a_center']
        anchor = self.top_k_dict[a_idx]
        _, a_g_feat, a_kpt, a_feat = self.load_desc_by_idxs([a_idx], unify_coord=True)
        a_kpt = (a_kpt - a_center) / self.scale
        a_kpt = transform_points(a_kpt, T_rnd)
        # load group data: idxs, states, sims, kpt, feat, score
        name_i = self.group_type.get_str()
        if self.expand_group:
            name_i = f'{self.group_type.get_str()}_expand'
        k_groups = item_i[name_i]
        k_idxs, k_states, k_kpt, k_feat, k_score = dict(), dict(), dict(), dict(), dict()
        for gp in k_groups:
            for i in gp['idx_in_gp']:
                if i in k_idxs:
                    continue
                k_idxs[i] = anchor['top_k'][i]  # int
                k_states[i] = anchor['state'][i]  # bool
                _, k_g_feat, kk, kf = self.load_desc_by_idxs([k_idxs[i]], unify_coord=True)
                k_sim_i = (a_g_feat @ k_g_feat.T + 1) / 2  # float
                k_kpt[i], k_feat[i] = kk[0], kf[0]  # m x 3, m x d
                k_score[i] = np.ones((k_kpt[i].shape[0], 1)) * k_sim_i  # m x 1
        # group ids
        gp_ids = list(range(len(k_groups)))
        if not self.dataset.data_cfg['is_test_dataset']:  # for training
            random.shuffle(gp_ids)
            n_pos, n_pos_train = 0, num_gp_train // 2
            n_neg, n_neg_train = 0, num_gp_train - n_pos_train
            tmp_gp_ids = []
            for gp_id in gp_ids:
                gp = k_groups[gp_id]
                if n_pos == n_pos_train and n_neg == n_neg_train:
                    break
                if gp['state'] is True:
                    if n_pos < n_pos_train:
                        n_pos = n_pos + 1
                        tmp_gp_ids.append(gp_id)
                else:
                    if n_neg < n_neg_train:
                        n_neg = n_neg + 1
                        tmp_gp_ids.append(gp_id)
            gp_ids = tmp_gp_ids
        # get data by group id
        gp_data = []
        for gp_id in gp_ids:
            gp = k_groups[gp_id]
            k_idxs_i, k_states_i, k_kpt_i, k_feat_i, k_score_i = [], [], [], [], []
            for j in gp['idx_in_gp']:
                k_idxs_i.append(k_idxs[j])
                k_states_i.append(k_states[j])
                k_kpt_i.append(k_kpt[j])
                k_feat_i.append(k_feat[j])
                k_score_i.append(k_score[j])
            k_kpt_i = np.stack(k_kpt_i, axis=0)  # ki x m x 3
            k_feat_i = np.stack(k_feat_i, axis=0)  # ki x m x d
            k_score_i = np.stack(k_score_i, axis=0)  # ki x m x 1
            # k kpt scale / centralization
            k_kpt_i = (k_kpt_i - gp['k_center']) / self.scale
            # gather kpt / feat / score according to center idxs
            if len(gp['idx_in_gp']) > 1:
                k_kfs = np.concatenate([k_kpt_i, k_feat_i, k_score_i], axis=-1)  # ki x m x (3+d+1)
                k_kfs = torch.from_numpy(k_kfs).to(device, non_blocking=True).float()  # ki x m x (3+d+1)
                k_kfs = k_kfs.view(1, -1, k_kfs.shape[-1])  # 1 x * x (3+d+1)
                center_idx = torch.from_numpy(gp['center_idx']).to(device, non_blocking=True)  # 1 x m
                k_kfs = self.gather_feats(k_kfs, center_idx).cpu().numpy()
                k_kpt_i, k_feat_i, k_score_i = k_kfs[..., :3], k_kfs[..., 3:(3+k_feat_i.shape[-1])], k_kfs[..., (3+k_feat_i.shape[-1]):]
            k_kpt_i, k_feat_i, k_score_i = k_kpt_i.squeeze(0), k_feat_i.squeeze(0), k_score_i.squeeze(0)
            gp_data_i = {
                'state': gp['state'],  # true / false
                'overlap': gp['overlap'],  # float
                'a_idx': a_idx,  # 1 x 1
                'a_kpt': a_kpt[0],  # m x 3
                'a_feat': a_feat[0],  # m x d
                'T_a': T_a,  # 4 x 4
                'k_idxs': k_idxs_i,  # list of int
                'k_states': k_states_i,  # list of bool
                'k_kpt': k_kpt_i,  # m x 3
                'k_feat': k_feat_i,  # m x d
                'k_score': k_score_i,  # m x 1
                'T_k': gp['T_k'],  # 4 x 4
                'T_gt': gp['T_gt'],  # 4 x 4
                'pairs': gp['pairs'],  # * x 2
                'unpair0': gp['unpair0'],  # *
                'unpair1': gp['unpair1'],  # *
                'matches0': gp['matches0'],  # m
            }
            # for debug
            show_pic = False
            if show_pic:
                from utils.draw_result import draw_pc_pps, draw_two_pc
                # coarse pps
                pairs = gp['pairs']
                title = f'a_idx: {a_idx}, gp_type: {name_i}, gp_id: {gp_id}, \n k_idxs: {k_idxs_i}, pps: {pairs.shape[0]}'
                transformed_a_kpt = transform_points(a_kpt[0], gp['T_gt'])
                if len(pairs) > 0:
                    a_kpt_ms, k_kpt_ms = a_kpt[0][pairs[:, 0]], k_kpt_i[pairs[:, 1]]
                    transformed_a_kpt_ms = transform_points(a_kpt_ms, gp['T_gt'])
                    draw_pc_pps(transformed_a_kpt, transformed_a_kpt_ms, k_kpt_i, k_kpt_ms, k_states_i,
                                title=title, offset_x=2.5)
                else:
                    draw_two_pc(transformed_a_kpt, k_kpt_i, title=title)
            gp_data.append(gp_data_i)
        return gp_data

    def parse_batch_data(self, in_batch_data, device, num_gp_train=4):
        gp_states, gp_overlaps, a_idx, k_idxs, k_states, k_sims = [], [], [], [], [], []
        a_kpt, a_feat, k_kpt, k_feat, k_score = [], [], [], [], []
        T_a, T_k, T_gt = [], [], []
        pairs, unpair0, unpair1, matches0 = [], [] , [], []
        for item_i in in_batch_data:
            if item_i is None:
                continue
            gp_data = self.parse_group_data(item_i, device, num_gp_train)
            for gpd in gp_data:
                gp_states.append(gpd['state'])  # bool
                gp_overlaps.append(gpd['overlap'])  # float
                a_idx.append(gpd['a_idx'])  # int
                k_idxs.append(gpd['k_idxs'])  # list of int
                k_states.append(gpd['k_states'])  # list of bool
                a_kpt.append(gpd['a_kpt'][None, ...])  # 1 x m x 3
                a_feat.append(gpd['a_feat'][None, ...])  # 1 x m x d
                T_a.append(gpd['T_a'])  # 4 x 4
                k_kpt.append(gpd['k_kpt'][None, ...])  # 1 x m x 3
                k_feat.append(gpd['k_feat'][None, ...])  # 1 x m x d
                k_score.append(gpd['k_score'][None, ...])  # 1 x m x 1
                T_k.append(gpd['T_k'])  # 4 x 4
                T_gt.append(gpd['T_gt'])  # 4 x 4
                pairs.append(gpd['pairs'])  # * x 2
                unpair0.append(gpd['unpair0'])  # *
                unpair1.append(gpd['unpair1'])  # *
                matches0.append(gpd['matches0'][None, ...])  # 1 x m
        # cat & tensor
        if len(a_idx) == 0:
            return None
        a_kpt = torch.from_numpy(np.concatenate(a_kpt, axis=0)).to(device, non_blocking=True)  # b x m x 3
        a_feat = torch.from_numpy(np.concatenate(a_feat, axis=0)).to(device, non_blocking=True)  # b x m x d
        k_kpt = torch.from_numpy(np.concatenate(k_kpt, axis=0)).to(device, non_blocking=True)  # b x m x 3
        k_feat = torch.from_numpy(np.concatenate(k_feat, axis=0)).to(device, non_blocking=True)  # b x m x d
        k_score = torch.from_numpy(np.concatenate(k_score, axis=0)).to(device, non_blocking=True)  # b x m x 1
        return {
            'gp_states': gp_states,  # bool
            'gp_overlaps': gp_overlaps,  # float
            'a_idx': a_idx,  # list of int
            'k_idxs': k_idxs,  # list of list of int
            'k_states': k_states,  # list of list of bool
            'a_kpt': a_kpt.float(),  # tensor, b x m x 3
            'a_feat': a_feat.float(),  # tensor, b x m x d
            'T_a': T_a,  # list of 4 x 4
            'k_kpt': k_kpt.float(),  # tensor, b x m x 3
            'k_feat': k_feat.float(),  # tensor, b x m x d
            'k_score': k_score.float(),  # tensor, b x m x 1
            'T_k': T_k,  # list of 4 x 4
            'T_gt': T_gt,  # list of 4 x 4
            'pairs': pairs, # list of array(* x 2)
            'unpair0': unpair0, # list of array(*)
            'unpair1': unpair1, # list of array(*)
            'matches0': matches0  # b x m
        }

    @staticmethod
    def sample_points(kpts, num_sample, correlation=None, sample_type=SamplingType.RANDOM):
        """ kpts: Tensor, b x m x 3, num_sample: num of points to sample, correlation: Tensor, b x m
            return: center_idxs, b x num_sample
        """
        center_idxs = None
        if sample_type == SamplingType.FPS:
            center_idxs = pointops.furthestsampling(kpts, num_sample)  # b x num_sample, num_sample < m
        elif sample_type == SamplingType.RANDOM:
            center_idxs = []
            for i in range(kpts.shape[0]):
                idxs = torch.randperm(kpts.shape[1]).to(kpts.device)[:num_sample][None, ...]  # 1 x num_sample
                center_idxs.append(idxs)
            center_idxs = torch.cat(center_idxs, dim=0).int()  # b x num_sample
        elif sample_type == SamplingType.CORRELATION:
            sorted_ind = torch.sort(correlation, dim=-1)[1].int()  # b x m, in increasing order
            num_weak, num_sample_weak, num_sample_strong = kpts.shape[1] // 5, 0, num_sample
            # weak
            # weak_indices = sorted_ind[:, :num_weak]
            # select_idxs = list(range(weak_indices.shape[-1]))
            # random.shuffle(select_idxs)
            # select_idxs = torch.IntTensor(select_idxs[:num_sample_weak]).to(sorted_ind.device)
            # weak_indices = torch.index_select(weak_indices, dim=-1, index=select_idxs)  # b x *
            # strong
            strong_indices = sorted_ind[:, num_weak:]
            select_idxs = list(range(strong_indices.shape[-1]))
            random.shuffle(select_idxs)
            select_idxs = torch.IntTensor(select_idxs[:num_sample_strong]).to(sorted_ind.device)
            strong_indices = torch.index_select(strong_indices, dim=-1, index=select_idxs)  # b x num_sample_strong
            #center_idxs = torch.cat([strong_indices], dim=-1)  # b x num_sample_strong
            center_idxs = strong_indices
        else:
            assert 'Invalid sampling type'
        return center_idxs

    @staticmethod
    def gather_feats(feats, idxs):
        """ feats: Tensor, b x m x d, idxs: Tensor, b x num_sample
            return: new feats, b x num_sample x d
        """
        feats = feats.transpose(1, 2).contiguous()  # b x d x m
        feats = pointops.gathering(feats, idxs).transpose(1, 2).contiguous()  # b x num_sample x d
        return feats

    @staticmethod
    def knn_points(query_kpt, ref_kpt, k, device, radius=None):
        """ query_kpt: b x m x 3, ref_kpt: b x n x 3
            return: knn_idxs, b x m x k
        """
        query_kpt = torch.from_numpy(query_kpt).to(device, non_blocking=True).float()
        ref_kpt = torch.from_numpy(ref_kpt).to(device, non_blocking=True).float()
        if radius is None:
            knn_idxs = pointops.knnquery(k, ref_kpt, query_kpt)  # b x m x k
        else:
            knn_idxs = pointops.ballquery(radius, k, ref_kpt, query_kpt)  # b x m x k
        return knn_idxs

    @staticmethod
    def find_pps(src_kpt, tgt_kpt, T_ts, device, pps_thre=1.0):
        src_kpt = transform_points(src_kpt, T_ts)  # tgt <-> trans_src = T_ts * src
        src_kpt = torch.from_numpy(src_kpt).to(device, non_blocking=True)  # m x 3
        tgt_kpt = torch.from_numpy(tgt_kpt).to(device, non_blocking=True)  # m x 3
        dist = torch.norm((src_kpt[:, None, :] - tgt_kpt[None, :, :]), dim=-1).cpu().numpy()
        argmin_of_0_in_1 = np.argmin(dist, axis=1)
        argmin_of_1_in_0 = np.argmin(dist, axis=0)
        tgt_kpt = tgt_kpt.cpu().numpy()
        overlap, pairs = 0, []
        matches0 = -np.ones(src_kpt.shape[0], dtype=int)
        matches1 = -np.ones(tgt_kpt.shape[0], dtype=int)
        for j in range(argmin_of_0_in_1.shape[0]):
            if dist[j, argmin_of_0_in_1[j]] < pps_thre * 2:
                overlap += 1
            if j == argmin_of_1_in_0[argmin_of_0_in_1[j]]:
                if dist[j, argmin_of_0_in_1[j]] < pps_thre:
                    pairs.append([j, argmin_of_0_in_1[j]])
                    matches0[j] = argmin_of_0_in_1[j]
                    matches1[argmin_of_0_in_1[j]] = j
        unpair0 = np.where(matches0 == -1)[0]
        unpair1 = np.where(matches1 == -1)[0]
        pairs = np.array(pairs)
        overlap = overlap / src_kpt.shape[0]
        return overlap, pairs, unpair0, unpair1, matches0

    def make_group(self, a_idx, a_kpt, T_a, gp_type, k_idx_in_dict, device,
                   sample_type=SamplingType.RANDOM, expand_gp=False, pps_thre=1.0, show_pic=False):
        """ groups: k indices in groups, k_idxs """
        # init top k indices
        k_idxs = []
        anchor = self.top_k_dict[a_idx]
        for i in k_idx_in_dict:
            k_idxs.append(anchor['top_k'][i])
        all_pos, _, all_kpt, _ = self.load_desc_by_idxs(anchor['top_k'], unify_coord=True)
        k_pos, _, k_kpt, k_feat = self.load_desc_by_idxs(k_idxs, unify_coord=True)
        # init groups
        groups, expand_thre = [], 30.0
        if gp_type == GroupType.FOR_TRAINING:  # for training
            groups = divide_topk_into_groups(k_pos, expand_thre)
            extra_groups = []
            for idxs_in_gp in groups:
                for j in range(len(idxs_in_gp)):
                    group_j = []
                    for k in range(j, len(idxs_in_gp)):
                        group_j.append(idxs_in_gp[k])
                        extra_groups.append(group_j.copy())
                        if len(group_j) == self.top_k:
                            break
            groups = extra_groups
        elif gp_type == GroupType.ONLY_TOP_1:
            groups = [[0]]
        elif gp_type == GroupType.INDIVIDUAL_TOP_K:
            for j in range(len(k_idx_in_dict)):
                groups.append([j])
        elif gp_type == GroupType.WHOLE_TOP_K:
            groups = [list(range(len(k_idx_in_dict)))]
        else:  # GroupType.GROUP_TOP_K
            groups = divide_topk_into_groups(k_pos, expand_thre)
        tmp_groups = []
        for gp in groups:
            tmp_gp = []
            for i in gp:
                tmp_gp.append(k_idx_in_dict[i])
            tmp_groups.append(tmp_gp)
        groups = tmp_groups
        # expand group, FIXME: init groups should be reserved for training!
        if (gp_type == GroupType.GROUP_TOP_K or gp_type == GroupType.FOR_TRAINING) and expand_gp:
            extra_groups = []
            for gp_id in range(len(groups)):
                extra_gp = []
                for i in groups[gp_id]:
                    for j in range(len(anchor['top_k'])):
                        if j in groups[gp_id] or j in extra_gp:
                            continue
                        pos_i, pos_j = all_pos[i], all_pos[j]
                        dist_ij = np.linalg.norm(pos_i - pos_j)
                        if dist_ij < expand_thre:
                            extra_gp.append(j)
                extra_groups.append(groups[gp_id] + extra_gp)
            if gp_type == GroupType.FOR_TRAINING:
                groups = groups + extra_groups
            else:
                groups = extra_groups
        if gp_type == GroupType.FOR_TRAINING:
            groups = list(np.random.choice(groups, 300, replace=False))

        # for debug: show group data
        if show_pic and False:
            from utils.draw_result import draw_pc_pps, draw_two_pc
            title = f'a_idx: {a_idx}'
            transformed_a_kpt = transform_points(a_kpt, T_a)
            draw_two_pc(transformed_a_kpt, all_kpt, use_diff_center=False, title=title)
        
        # make groups
        k_groups = []
        for gp_id in range(len(groups)):
            k_idxs_i, k_states_i, k_kpt_i = [], [], []
            for j in groups[gp_id]:
                k_idxs_i.append(anchor['top_k'][j])
                k_states_i.append(anchor['state'][j])
                k_kpt_i.append(all_kpt[j])
            k_kpt_i = np.stack(k_kpt_i, axis=0)  # ki x n x 3
            # merge top k and normalize key points
            k_kpt_i, k_center_i = merge_topk(k_kpt_i, self.scale)  # ki x n x 3, 1 x 3
            # judge whether current group is positive / negative
            is_positive = len([x for x in k_states_i if x == 1]) > 0
            # gt transformation
            T_k_i = np.block([[np.eye(3) * self.scale, k_center_i.T],
                              [np.zeros([1, 3]), np.ones([1, 1])]])
            T_gt_i = np.matmul(np.linalg.inv(T_k_i), T_a)  # P_k = T_gt * P_a
            # furthest point sample
            if k_kpt_i.shape[0] > 1:
                k_kpt_i = k_kpt_i.reshape(1, -1, 3)  # 1 x * x 3
                k_kpt_i = torch.from_numpy(k_kpt_i).to(device, non_blocking=True).float()  # 1 x * x 3
                center_idx = self.sample_points(k_kpt_i, a_kpt.shape[0], None, sample_type=sample_type)  # 1 x m
                k_kpt_i = self.gather_feats(k_kpt_i, center_idx)  # m x 3
                center_idx = center_idx.cpu().numpy()
                k_kpt_i = k_kpt_i.cpu().numpy()
            else:
                center_idx = None
            k_kpt_i = k_kpt_i.reshape(-1, k_kpt_i.shape[-1])  # m x 3
            # find point pairs
            if is_positive:
                overlap, pairs, unpair0, unpair1, matches0 = self.find_pps(a_kpt, k_kpt_i, T_gt_i, device, pps_thre)
            else:
                overlap = 0.0
                pairs = np.array([])
                unpair0 = np.array([])
                unpair1 = np.array([])
                matches0 = np.array([])
            gp = {
                'state': is_positive,  # true / false
                'overlap': overlap,  # float
                'idx_in_gp': groups[gp_id],  # list of int
                'k_center': k_center_i,  # 1 x 3
                'center_idx': center_idx,  # 1 x m, it's None if len(idx_in_gp) == 1
                'T_k': T_k_i,  # 4 x 4
                'T_gt': T_gt_i,  # 4 x 4
                'pairs': pairs,  # * x 2, refer to sample tgt/k
                'unpair0': unpair0,  # *
                'unpair1': unpair1,  # *
                'matches0': matches0  # m
            }
            k_groups.append(gp)
            # for debug: show pps
            if show_pic and len(k_idxs_i) > 1:
                from utils.draw_result import draw_pc_pps, draw_two_pc
                # coarse pps
                title = f'a_idx: {a_idx}, gp_type: {gp_type.get_str()}, gp_id: {gp_id}, \n k_idxs: {k_idxs_i}, pps: {len(pairs)}'
                transformed_a_kpt = transform_points(a_kpt, T_gt_i)
                if len(pairs) > 0:
                    a_kpt_ms, k_kpt_ms = a_kpt[pairs[:, 0]], k_kpt_i[pairs[:, 1]]
                    transformed_a_kpt_ms = transform_points(a_kpt_ms, T_gt_i)
                    draw_pc_pps(transformed_a_kpt, transformed_a_kpt_ms, k_kpt_i, k_kpt_ms, k_states_i,
                                title=title, offset_x=2.5)
                else:
                    draw_two_pc(transformed_a_kpt, k_kpt_i, title=title)
        return k_groups

    def prepare_data(self, gp_types, device, pps_thre=1.0, show_pic=False):
        # save dir
        if self.__len__() > 0:
            dataset_type_str = 'test' if self.dataset.data_cfg['is_test_dataset'] else 'train'
            sample_type_str = self.sample_type.str()
            save_dir = os.path.join(self.dataset.pickle_dir(),
                                    f'pose_est_top{self.top_k}_{dataset_type_str}_{sample_type_str}')
            utils.util.check_makedirs(save_dir)
        # process
        pps_thre = pps_thre / self.scale
        for i in tqdm(range(self.__len__()), desc='Prepare Data for Pose Estimation'):
            # query and top k indices
            a_idx = self.top_k_dict_keys[i]
            if not self.dataset.data_cfg['is_test_dataset']:  # for training
                k_idx_in_dict = list(range(len(self.top_k_dict[a_idx]['top_k'])))
            else:  # for testing
                k_idx_in_dict = list(range(self.top_k))
            if len(k_idx_in_dict) < self.top_k:
                continue
            # load data (g_feat: k x d, l_feat: k x n x d, l_pos: k x n x 3, pos: k x 3)
            _, _, a_kpt, _ = self.load_desc_by_idxs([a_idx], unify_coord=True)
            # deal with anchor: normalize and rotate randomly
            a_kpt = a_kpt.reshape(-1, a_kpt.shape[-1])
            a_kpt, a_center = merge_topk(a_kpt, self.scale)
            T_rnd = self.make_random_transformation()
            a_kpt = transform_points(a_kpt, T_rnd)  # randomly transform the anchor cloud
            T_a = np.block([[np.eye(3) * self.scale, a_center.T],
                            [np.zeros([1, 3]), np.ones([1, 1])]])
            T_a = np.matmul(T_a, np.linalg.inv(T_rnd))  # T_a' = T_a * T_rnd.inv()
            item_i = {
                'a_idx': a_idx,  # int
                'a_center': a_center,  # 1 x 3
                'T_rnd': T_rnd,  # 4 x 4
                'T_a': T_a,  # 4 x 4, T_a' = T_a * T_rnd.inv()
            }
            # divide top k into groups, and expand each group
            for gp_type in gp_types:
                # make groups: no expansion
                k_groups = self.make_group(a_idx, a_kpt, T_a, gp_type, k_idx_in_dict,
                                           device, self.sample_type, False, pps_thre, show_pic)
                name_i = f'{gp_type.get_str()}'
                item_i[name_i] = k_groups
                # make groups: expansion
                k_groups = self.make_group(a_idx, a_kpt, T_a, gp_type, k_idx_in_dict,
                                           device, self.sample_type, True, pps_thre, show_pic)
                name_i = f'{gp_type.get_str()}_expand'
                item_i[name_i] = k_groups
            # save
            item_pkl = os.path.join(save_dir, f'{a_idx}.pickle')
            with open(item_pkl, 'wb') as handle:
                pickle.dump(item_i, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_top_k_index(self, query_trip_idx=-1, ref_trip_idx=-1, key_slice=None, use_rerank=True):
        dataset_type_str = 'test' if self.dataset.data_cfg['is_test_dataset'] else 'train'
        pkl_type = 'rerank' if use_rerank else 'init'
        if query_trip_idx == -1 or ref_trip_idx == -1:
            pkl_name = f"top_k_index_{dataset_type_str}_{pkl_type}.pickle"
        else:
            pkl_name = f"top_k_index_{dataset_type_str}_{query_trip_idx}_{ref_trip_idx}_{pkl_type}.pickle"
        top_k_pkl = os.path.join(self.dataset.desc_dir(self.pr_backbone), pkl_name)
        self.top_k_dict = None
        if os.path.exists(top_k_pkl):
            with open(top_k_pkl, 'rb') as handle:
                self.top_k_dict = pickle.load(handle)
                print('load top k index: ', top_k_pkl)
        else:
            print('Fail to load: ', top_k_pkl)
        self.top_k_dict_keys = [] if self.top_k_dict is None else list(self.top_k_dict.keys())
        if key_slice is not None:
            self.top_k_dict_keys = self.top_k_dict_keys[key_slice]


if __name__ == '__main__':
    # test manual gt
    a_kpt = np.random.randn(1, 128, 3)
    k_kpt = a_kpt
    a_desc = np.random.randn(1, 128, 32)
    k_desc = a_desc
    scale = 30.0
    a_kpt, a_center = merge_topk(a_kpt, scale)
    k_kpt, k_center = merge_topk(k_kpt, scale)
    a_kpt, k_kpt = a_kpt.reshape(-1, 3), k_kpt.reshape(-1, 3)
    a_desc, k_desc = a_desc.reshape(-1, 32), k_desc.reshape(-1, 32)
    rnd_R = random_rotation_matrix()
    T_rnd = np.block([[rnd_R, np.zeros([3, 1])],
                      [np.zeros([1, 3]), np.ones([1, 1])]])
    a_kpt = transform_points(a_kpt, rnd_R)  # randomly transform the query cloud
    T_a = np.block([[np.eye(3) * scale, a_center.T],
                    [np.zeros([1, 3]), np.ones([1, 1])]])
    T_k = np.block([[np.eye(3) * scale, k_center.T],
                    [np.zeros([1, 3]), np.ones([1, 1])]])
    T_gt = np.matmul(np.linalg.inv(T_k), np.matmul(T_a, np.linalg.inv(T_rnd)))  # P_k = T_gt * P_a
    k_kpt2 = transform_points(a_kpt, T_gt)
    diff = np.sum(np.sum(np.square(k_kpt2 - k_kpt), axis=-1))
    # estimate transformation by ransac
    from pose_estimation.pose_est_ransac import nn_match, estimate_pose_ransac
    k_kpt, k_desc = k_kpt.reshape(-1, k_kpt.shape[-1]), k_desc.reshape(-1, k_desc.shape[-1])
    pps = nn_match(a_desc, k_desc)  # * x 2
    pose_est_res = estimate_pose_ransac(a_kpt, a_desc, k_kpt, k_desc, pps, max_iter=1000)
    # data
    data_names = [ 'hankou',
                   #'campus'
                 ]
    group_types = [GroupType.ONLY_TOP_1,
                   GroupType.INDIVIDUAL_TOP_K,
                   GroupType.WHOLE_TOP_K,
                   GroupType.GROUP_TOP_K
                   ]
    sample_type = SamplingType.FPS
    pps_thre = 1.0
    for data_name in data_names:
        # # train, top 5
        t_dataset = PoseEstDataset(data_name, True, top_k=5, sample_type=sample_type)
        t_dataset.load_top_k_index(key_slice=slice(0, 1200, 1), use_rerank=False)
        t_dataset.prepare_data([GroupType.FOR_TRAINING], device='cuda:0', pps_thre=pps_thre, show_pic=False)

        # test, top 5 and top 1
        # query / ref trip pairs
        data_cfg = t_dataset.dataset.data_cfg
        query_ref_idxs = []
        for query_trip_idx in range(len(t_dataset.dataset.trip_names)):
            for ref_trip_idx in range(len(t_dataset.dataset.trip_names)):
                if query_trip_idx == ref_trip_idx:
                    continue
                if data_cfg['test_query_trips'] is not None and (
                        t_dataset.dataset.trip_names[query_trip_idx] not in data_cfg['test_query_trips']):
                    continue
                query_ref_idxs.append((query_trip_idx, ref_trip_idx))
        t_dataset = PoseEstDataset(data_name, False, top_k=5, sample_type=sample_type)
        for query_trip_idx, ref_trip_idx in query_ref_idxs:
            t_dataset.load_top_k_index(query_trip_idx, ref_trip_idx, key_slice=slice(0, 300, 1), use_rerank=False)
            t_dataset.prepare_data(group_types, device='cuda:0', pps_thre=pps_thre, show_pic=False)
