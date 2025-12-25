import copy
import os.path

import numpy as np
import torch.nn
from torch.utils import data
import torch.nn.functional as F

from datasets.scene_dataset import *
from utils.train_util import nn_dist


class RerankDataset(data.Dataset):
    """ Dataset for reranking the top k place recognition results """
    def __init__(self, name, for_training, pr_backbone, top_k=25, neigh_k=4, neigh_step=5, use_hard_pos_neg=False):
        super(RerankDataset, self).__init__()
        self.pr_backbone = pr_backbone
        self.dataset = SceneDataSet(name, for_training)
        self.dataset.load(query_trip_indices=None)
        self.feat_knn_dict, self.euc_knn_dict = None, None
        self.top_k_dict_keys, self.top_k = [], top_k
        self.neigh_k, self.neigh_step = neigh_k, neigh_step
        self.num_pos, self.num_neg = 2, 0
        self.use_hard_pos_neg = use_hard_pos_neg
        self.hard_pos_neg_dict = dict()  # key: anchor; value: { 'pos': list, 'neg': list }

    def __getitem__(self, index):
        # anchor
        a_idx = self.top_k_dict_keys[index]
        last_a_idxs = self.dataset.last_k(a_idx, k=self.neigh_k, interval=self.neigh_step)
        return last_a_idxs + [a_idx]  # sequence anchors

    def __len__(self):
        return len(self.top_k_dict_keys)

    def get_knn_tuple(self, idx, space_type='feat'):
        topk_idxs = []
        if space_type == 'feat':  # for anchor
            retrieved_idxs = np.array(self.feat_knn_dict[idx]['top_k'], dtype=int)
            retrieved_states = np.array(self.feat_knn_dict[idx]['state'], dtype=int)
            retrieved_states = np.array(retrieved_states == 1, dtype=int)
            if not self.dataset.data_cfg['is_test_dataset']:  # for training
                # select random top k for training
                n_search = 0
                gt_labels = []
                sample_indices = list(range(retrieved_idxs.shape[0]))
                while np.sum(gt_labels) == 0 or np.sum(gt_labels) == self.top_k:
                    n_search += 1
                    random.shuffle(sample_indices)
                    selected_indices = np.array(sample_indices[:self.top_k], dtype=int)
                    topk_idxs = retrieved_idxs[selected_indices]
                    gt_labels = retrieved_states[selected_indices]
                    if n_search == 1000:
                        break
            else:  # for testing
                topk_idxs = retrieved_idxs[:self.top_k]
        else:  # euc knn, for positive / negative
            if not self.dataset.data_cfg['is_test_dataset']:  # for training
                topk_idxs = np.random.choice(self.euc_knn_dict[idx]['euc_knn'], self.top_k, replace=False)
            else:
                topk_idxs = np.array(self.euc_knn_dict[idx]['euc_knn'][:self.top_k])
        # check
        if topk_idxs.shape[0] != self.top_k:
            return None
        # shuffle
        ranks = np.array(range(self.top_k), dtype=int)
        np.random.shuffle(ranks)
        topk_idxs = np.take_along_axis(topk_idxs, ranks, axis=-1)
        return topk_idxs

    def load_desc_by_idxs(self, idxs, device, use_local=False):
        idxs = list(idxs)
        # top k desc and pos
        global_descs, local_descs, local_poss, positions = [], [], [], []
        for idx in idxs:
            # global
            global_desc, position = self.dataset.get_g_desc(self.pr_backbone, idx), self.dataset.get_pos_xyz(idx)
            if global_desc is None:
                return None, None, None, None
            position = position.reshape(1, position.shape[-1])
            global_descs.append(global_desc)
            positions.append(position)
            # local
            if use_local:
                local_pos, local_desc = self.dataset.get_l_kpt_desc(self.pr_backbone, idx)
                local_pos = local_pos.reshape(1, local_pos.shape[0], local_pos.shape[1])
                local_desc = local_desc.reshape(1, local_desc.shape[0], local_desc.shape[1])
                local_descs.append(local_desc)
                local_poss.append(local_pos)
        global_descs = np.concatenate(global_descs, axis=0)
        positions = np.concatenate(positions, axis=0)
        if use_local:
            local_poss = np.concatenate(local_poss, axis=0)
            local_descs = np.concatenate(local_descs, axis=0)
        # global_descs: 1 x k x d, local_descs: 1 x k x n x d,
        # local_poss: 1 x k x n x 3, positions: 1 x k x 3
        # scale: 1 x k x 3, translation: 1 x k x 3
        g_desc, l_pos, l_desc, pos = self.np2torch_unsqueeze(global_descs, local_poss, local_descs, positions,
                                                             device, non_blocking=True)
        return pos, g_desc, l_pos, l_desc

    @staticmethod
    def np2torch_unsqueeze(g_desc, l_pos, l_desc, pos, device, non_blocking):
        # global
        g_desc = torch.from_numpy(g_desc).to(device, non_blocking=non_blocking)
        g_desc = g_desc.unsqueeze(0)  # 1 x 1 x d
        pos = torch.from_numpy(pos).to(device, non_blocking=non_blocking)
        pos = pos.unsqueeze(0)  # 1 x 1 x 3
        # local
        if len(l_desc):
            l_desc = torch.from_numpy(l_desc).to(device, non_blocking=non_blocking)
            l_desc = l_desc.unsqueeze(0)  # 1 x 1 x n x d
        if len(l_pos):
            l_pos = torch.from_numpy(l_pos).to(device, non_blocking=non_blocking)
            l_pos = l_pos.unsqueeze(0)  # 1 x 1 x n x 3
        return g_desc, l_pos, l_desc, pos

    def parse_batch_data(self, in_batch_data, device, ref_trip_idx=-1, use_local=False):
        batches = []
        for item_i in in_batch_data:
            if item_i is None:
                continue
            a_idxs = item_i
            # init anchor / top k for init eval
            init_data = self.get_data_for_init_eval(a_idxs[-1])

            # # seq top k: seq data for training / testing
            # seq_data = self.get_seq_top_k(a_idxs, k=self.top_k, device=device)
            # if seq_data is None and self.dataset.for_training():
            #     continue

            # seq P-GAT: seq data for training / testing
            if self.dataset.for_training():
                seq_data = self.get_seq_PGAT_train(a_idxs[-1], device=device)
            else:
                seq_data = self.get_seq_PGAT_test(a_idxs[-1], device=device)
            if seq_data is not None:
                # a
                a_pos, a_g_desc = seq_data['a'][0], seq_data['a'][1],
                a_pos = a_pos.repeat(len(seq_data['seq']), 1, 1)
                a_g_desc = a_g_desc.repeat(len(seq_data['seq']), 1, 1)
                seq_data['a'] = (a_pos, a_g_desc, None, None)
                # k
                k_pos, k_g_desc = [], []
                for i in range(len(seq_data['seq'])):
                    pos, g_desc = seq_data['seq'][i]['k'][0], seq_data['seq'][i]['k'][1]
                    k_pos.append(pos)
                    k_g_desc.append(g_desc)
                    seq_data['seq'][i]['k'] = None
                k_pos = torch.cat(k_pos, dim=0)
                k_g_desc = torch.cat(k_g_desc, dim=0)
                seq_data['k'] = (k_pos, k_g_desc, None, None)
            batch = {
                'init_eval': init_data,
                'train_test': seq_data
            }
            batches.append(batch)
        return batches

    def expand_top_k(self, a_idx, top_k_idx, device, reserve_ratio=0.5, use_local=False):
        """ discard some unreliable items, and expand the same number of items
            from the map according to their positions
            top_k_idx: indices of ranked top k
        """
        # num to expand
        n_total = len(top_k_idx)
        n_reserve = int(n_total * reserve_ratio)
        n_expand = n_total - n_reserve
        assert (n_expand > 0) and 'reserve ratio for expansion is too large!'
        # database
        database_indices = self.feat_knn_dict[a_idx]['top_k']
        database_labels = np.array(np.array(self.feat_knn_dict[a_idx]['state'], dtype=int) > 0, dtype=int)
        database_tree = self.dataset.construct_position_tree(database_indices)
        # find neighbors
        neighbor_dict = dict()
        top_k_idx = top_k_idx[:n_reserve]
        for k_idx in top_k_idx:
            k_pos = self.pos_dict[k_idx]
            neighbor_dict[k_idx] = database_tree.query(k_pos.reshape(1, -1), k=n_total)
        # expand
        selected_indices, count = [], 0
        while len(selected_indices) < n_total:
            round_count = count // len(top_k_idx)
            k_idx = top_k_idx[count % len(top_k_idx)]
            kk_dist, index = neighbor_dict[k_idx][0], neighbor_dict[k_idx][1]
            kk_dist, index = kk_dist[0][round_count], index[0][round_count]
            count = count + 1
            if index in selected_indices:
                continue
            selected_indices.append(index)
        assert len(selected_indices) == n_total
        # init order
        selected_indices = np.array(selected_indices, dtype=int)
        sorted_indices = np.argsort(selected_indices, axis=-1)
        selected_indices = np.take_along_axis(selected_indices, sorted_indices, axis=-1)
        new_pn_idx = np.array(database_indices, dtype=int)[selected_indices]
        pn_labels = np.array(database_labels, dtype=int)[selected_indices]
        pn_labels = torch.from_numpy(pn_labels).to(device, non_blocking=True)
        # anchor
        a_pos, a_g_desc, a_l_pos, a_l_desc = self.load_desc_by_idxs([a_idx], device, use_local)
        pn_pos, pn_g_descs, pn_l_pos, pn_l_descs = self.load_desc_by_idxs(new_pn_idx, device, use_local)
        a_g_desc_copy = a_g_desc.repeat(1, pn_g_descs.shape[1], 1).view(-1, pn_g_descs.shape[-1])
        origin_sim = F.cosine_similarity(a_g_desc_copy, pn_g_descs, dim=-1)
        return {
            'apn_idx': (a_idx, new_pn_idx),
            'a': (a_pos, a_g_desc, a_l_pos, a_l_desc),
            'pn': (pn_pos, pn_g_descs, pn_l_pos, pn_l_descs),
            'gt_label': pn_labels,
            'origin_sim': origin_sim
        }

    def load_top_k_index(self, query_trip_idx=-1, ref_trip_idx=-1, key_slice=None, load_top_k_pkl=True):
        if load_top_k_pkl:
            # top k in feature space
            dataset_type_str = 'test' if self.dataset.data_cfg['is_test_dataset'] else 'train'
            pkl_name = f"top_k_index_{dataset_type_str}_{query_trip_idx}_{ref_trip_idx}_init.pickle"
            top_k_pkl = os.path.join(self.dataset.desc_dir(self.pr_backbone), pkl_name)
            self.feat_knn_dict = None
            if os.path.exists(top_k_pkl):
                with open(top_k_pkl, 'rb') as handle:
                    self.feat_knn_dict = pickle.load(handle)
                    print('load top k in feature space: ', top_k_pkl)
            else:
                print('Fail to load top k in euclidean space: ', top_k_pkl)

            # top k in euclidean space
            pkl_name = f"top_k_index_{dataset_type_str}_{ref_trip_idx}_init.pickle"
            top_k_pkl = os.path.join(self.dataset.euc_knn_dir(), pkl_name)
            self.euc_knn_dict = None
            if os.path.exists(top_k_pkl):
                with open(top_k_pkl, 'rb') as handle:
                    self.euc_knn_dict = pickle.load(handle)
                    print('load top k in euclidean space: ', top_k_pkl)
            else:
                print('Fail to load top k in euclidean space: ', top_k_pkl)

        self.top_k_dict_keys = [] if self.feat_knn_dict is None else list(self.feat_knn_dict.keys())
        if key_slice is not None:
            self.top_k_dict_keys = self.top_k_dict_keys[key_slice]

    def load_related_data(self, load_pc=False, downsample_pc=False, normalize_pc=False, create_fpfh=False,
                          load_g_desc=True, load_l_desc=False, k=-1):
        # idxs to load
        idxs_to_load = set()
        if load_pc or load_g_desc or load_l_desc or create_fpfh:
            for key in self.top_k_dict_keys:
                real_k = len(self.feat_knn_dict[key]['top_k']) if k == -1 else k
                idxs_to_load = idxs_to_load | set(self.feat_knn_dict[key]['top_k'][:real_k])
        idxs_to_load = list(idxs_to_load)
        cache_size = min(len(idxs_to_load) * self.top_k, 50000)
        self.dataset.set_cache_size(cache_size)
        # load related data
        if load_pc:
            self.dataset.get_pcs(idxs_to_load, downsample_pc, normalize_pc)
            if create_fpfh:
                self.dataset.get_fpfhs(idxs_to_load)
        if load_g_desc:
            self.dataset.get_g_descs(self.pr_backbone, idxs_to_load)
        if load_l_desc:
            self.dataset.get_l_kpts_descs(self.pr_backbone, idxs_to_load)

    def get_data_for_init_eval(self, a_idx):
        init_data = {
            'a_idx': a_idx,
            'top_k_idx': np.array(self.feat_knn_dict[a_idx]['top_k'][:self.top_k], dtype=int),
            'top_k_label': np.array(self.feat_knn_dict[a_idx]['state'][:self.top_k], dtype=int)
        }
        _, a_g_desc, _, _ = self.load_desc_by_idxs([a_idx], 'cpu')
        _, k_g_desc, _, _ = self.load_desc_by_idxs(init_data['top_k_idx'], 'cpu')
        a_g_desc = a_g_desc.repeat(1, k_g_desc.shape[1], 1)
        init_data['top_k_sim'] = F.cosine_similarity(a_g_desc, k_g_desc, dim=-1)[0].cpu().numpy()
        return init_data

    def get_seq_PGAT_train(self, a_idx, device):
        batch = dict()
        # anchor
        last_a_idxs = self.dataset.last_k(a_idx, self.neigh_k//2, self.neigh_step)
        next_a_idxs = self.dataset.next_k(a_idx, self.neigh_k - self.neigh_k // 2, self.neigh_step)
        a_idxs = last_a_idxs + [a_idx] + next_a_idxs
        if len(a_idxs) != self.neigh_k + 1:
            return None
        a_pos, a_g_desc, a_l_kpt, a_l_desc = self.load_desc_by_idxs(a_idxs, device)
        batch['a_idx'] = np.array(a_idxs, dtype=int)  # array: a
        batch['a'] = (a_pos, a_g_desc, a_l_kpt, a_l_desc)  # tensor
        # pos start/end
        positives = set()
        for a_idx in a_idxs:
            a_tuple = self.dataset.get_tuple(a_idx, ref_trip_idx=self.dataset.current_ref_trip_idx,
                                             skip_trip_itself=True)
            positives = positives | set(a_tuple.positive_indices)
        # pos starts
        pos_starts = list(positives)
        for p in positives:
            pos_starts = pos_starts + list(range(p - self.neigh_k * self.neigh_step, p))
        pos_starts = list(set(pos_starts))
        np.random.shuffle(pos_starts)
        # pos end indices
        pos_idxs = []
        ref_sample_indices = self.dataset.sample_indices[self.dataset.current_ref_trip_idx]
        for pos_start in pos_starts:
            if pos_start < ref_sample_indices[0] or pos_start+(self.neigh_k+1)*self.neigh_step >= ref_sample_indices[-1]:
                continue
            if len(pos_idxs) == self.num_pos:
                break
            pos_idxs.append(pos_start + self.neigh_k//2 * self.neigh_step)
        if len(pos_idxs) < self.num_pos:
            return None
        # neg starts
        neg_starts = list(set(ref_sample_indices) - set(pos_starts))
        np.random.shuffle(neg_starts)
        # neg end indices
        neg_idxs = []
        for neg_start in neg_starts:
            if neg_start < ref_sample_indices[0] or neg_start+(self.neigh_k+1)*self.neigh_step >= ref_sample_indices[-1]:
                continue
            if len(neg_idxs) == self.num_neg:
                break
            neg_idxs.append(neg_start + self.neigh_k//2 * self.neigh_step)
        # data
        search_radius_pos = self.dataset.data_cfg['search_radius_pos']
        search_radius_neg = self.dataset.data_cfg['search_radius_neg']
        batch['top_k_idx'] = np.array(pos_idxs + neg_idxs, dtype=int)
        batch['top_k_label'] = np.array([1]*len(pos_idxs)+[0]*len(neg_idxs), dtype=int)
        batch['top_k_label'] = torch.from_numpy(batch['top_k_label']).to(device)
        k_pos, k_g_desc, k_l_kpt, k_l_desc = self.load_desc_by_idxs(pos_idxs + neg_idxs, device)
        tmp_a_g_desc = a_g_desc[:, len(last_a_idxs):len(last_a_idxs)+1, :]
        batch['init_sim'] = F.cosine_similarity(tmp_a_g_desc, k_g_desc, dim=-1)
        batch['seq'] = []
        for i in range(len(batch['top_k_idx'])):
            k_idx = batch['top_k_idx'][i]
            last_k_idxs = self.dataset.last_k(k_idx, self.neigh_k//2, self.neigh_step)
            next_k_idxs = self.dataset.next_k(k_idx, self.neigh_k - self.neigh_k // 2, self.neigh_step)
            if len(last_k_idxs) + len(next_k_idxs) < self.neigh_k:
                return None
            seq_k_idxs = last_k_idxs + [k_idx] + next_k_idxs
            k_pos, k_g_desc, k_l_kpt, k_l_desc = self.load_desc_by_idxs(seq_k_idxs, device)
            # state: pos / neg ?
            ak_dist = torch.sum((a_pos.permute(1, 0, 2) - k_pos) ** 2, dim=-1) ** 0.5
            states = torch.ones(ak_dist.size(), dtype=torch.long).to(ak_dist.device) * -1  # -1 means unknown
            states = torch.where(ak_dist < search_radius_pos, torch.ones_like(states), states)
            states = torch.where(ak_dist > search_radius_neg, torch.zeros_like(states), states)
            seq = {
                'k_idx': np.array(seq_k_idxs, dtype=int),  # array: k
                'state': states,  # tensor: a x k
                'k': (k_pos, k_g_desc, k_l_kpt, k_l_desc),  # tensor
            }
            batch['seq'].append(seq)
        return batch

    def get_seq_PGAT_test(self, a_idx, device):
        # anchor
        last_a_idxs = self.dataset.last_k(a_idx, self.neigh_k // 2, self.neigh_step)
        next_a_idxs = self.dataset.next_k(a_idx, self.neigh_k - self.neigh_k // 2, self.neigh_step)
        a_idxs = last_a_idxs + [a_idx] + next_a_idxs
        if len(a_idxs) != self.neigh_k + 1:
            return None
        a_pos, a_g_desc, a_l_kpt, a_l_desc = self.load_desc_by_idxs(a_idxs, device)
        # top k
        k_idxs = self.feat_knn_dict[a_idxs[len(last_a_idxs)]]['top_k'][:self.top_k]
        # share info
        batch = dict()
        batch['a_idx'] = np.array(a_idxs, dtype=int)  # array: a
        batch['a'] = (a_pos, a_g_desc, a_l_kpt, a_l_desc)  # tensor
        batch['top_k_idx'] = np.array(k_idxs, dtype=int)
        batch['top_k_label'] = np.array(self.feat_knn_dict[a_idxs[len(last_a_idxs)]]['state'][:self.top_k], dtype=int)
        batch['top_k_label'] = torch.from_numpy(batch['top_k_label']).to(device)
        k_pos, k_g_desc, k_l_kpt, k_l_desc = self.load_desc_by_idxs(k_idxs, device)
        tmp_a_g_desc = a_g_desc[:, len(last_a_idxs):len(last_a_idxs)+1, :]
        batch['init_sim'] = F.cosine_similarity(tmp_a_g_desc, k_g_desc, dim=-1)
        # seq
        search_radius_pos = self.dataset.data_cfg['search_radius_pos']
        search_radius_neg = self.dataset.data_cfg['search_radius_neg']
        batch['seq'] = []
        for i in range(len(k_idxs)):
            k_idx = k_idxs[i]
            last_k_idxs = self.dataset.last_k(k_idx, self.neigh_k // 2, self.neigh_step)
            next_k_idxs = self.dataset.next_k(k_idx, self.neigh_k - self.neigh_k // 2, self.neigh_step)
            if len(last_k_idxs) + len(next_k_idxs) < self.neigh_k:
                return None
            seq_k_idxs = last_k_idxs + [k_idx] + next_k_idxs
            k_pos, k_g_desc, k_l_kpt, k_l_desc = self.load_desc_by_idxs(seq_k_idxs, device)
            # state: pos / neg ?
            ak_dist = torch.sum((a_pos.permute(1, 0, 2) - k_pos) ** 2, dim=-1) ** 0.5
            states = torch.ones(ak_dist.size(), dtype=torch.long).to(ak_dist.device) * -1  # -1 means unknown
            states = torch.where(ak_dist < search_radius_pos, torch.ones_like(states), states)
            states = torch.where(ak_dist > search_radius_neg, torch.zeros_like(states), states)
            seq = {
                'k_idx': np.array(seq_k_idxs, dtype=int),  # array: k
                'state': states,  # tensor: a x k
                'k': (k_pos, k_g_desc, k_l_kpt, k_l_desc),  # tensor
            }
            batch['seq'].append(seq)
        return batch

    def get_seq_PGAT_subgraph(self, start_idx, trip_idx, device):
        idxs_in_trip = self.dataset.sample_indices[trip_idx]
        if start_idx not in idxs_in_trip:
            return None
        seq_idxs = self.dataset.next_k(start_idx, self.neigh_k, self.neigh_step)
        if len(seq_idxs) < self.neigh_k:
            return None
        seq_idxs = [start_idx] + seq_idxs
        pos, g_desc, l_kpt, l_desc = self.load_desc_by_idxs(seq_idxs, device)
        return {
            'seq_idx': seq_idxs,
            'data': (pos, g_desc, l_kpt, l_desc)
        }

    def get_seq_top_k(self, a_idxs, k, device):
        """ get top k of one sequence by voting """
        valid_a_idxs, k_idxs = [], []
        for a_idx in a_idxs:
            if a_idx not in self.feat_knn_dict:
                continue
            valid_a_idxs.append(a_idx)
            current_k_idxs = self.feat_knn_dict[a_idx]['top_k'][:k]
            k_idxs = k_idxs + current_k_idxs
        k_idxs = list(set(k_idxs))
        if len(k_idxs) < self.neigh_k + 1 or len(valid_a_idxs) < self.neigh_k+1:
            return None
        # vote by score
        np.random.shuffle(k_idxs)
        a_pos, a_g_desc, a_l_kpt, a_l_desc = self.load_desc_by_idxs(valid_a_idxs, device)
        k_pos, k_g_desc, k_l_kpt, k_l_desc = self.load_desc_by_idxs(k_idxs, device)
        origin_sim = (F.cosine_similarity(a_g_desc.unsqueeze(2), k_g_desc.unsqueeze(1), dim=-1).squeeze(0) + 1) / 2  # Na x Nk
        vote_type = 'max'  # weight, max, none
        if vote_type == 'weight':
            weight = F.softmax(origin_sim, dim=0)
            new_sim = torch.sum(torch.mul(origin_sim, weight), dim=0)  # Nk
        elif vote_type == 'max':  # max sim
            new_sim = torch.max(origin_sim, dim=0)[0]  # Nk
        else:
            new_sim = origin_sim[-1]
        _, index = torch.topk(new_sim, k=self.neigh_k + 1, sorted=False)
        k_idxs = np.array(k_idxs)[index.cpu().numpy()]
        origin_sim = torch.index_select(origin_sim, dim=1, index=index)  # Na x k
        k_pos = torch.index_select(k_pos, dim=1, index=index)
        k_g_desc = torch.index_select(k_g_desc, dim=1, index=index)
        if k_l_kpt and k_l_desc:
            k_l_kpt = torch.index_select(k_l_kpt, dim=1, index=index)
            k_l_desc = torch.index_select(k_l_desc, dim=1, index=index)
        # dist
        ak_dist = torch.sum((a_pos.permute(1, 0, 2) - k_pos)**2, dim=-1) ** 0.5
        # state: pos / neg ?
        search_radius_pos = self.dataset.data_cfg['search_radius_pos']
        search_radius_neg = self.dataset.data_cfg['search_radius_neg']
        if self.dataset.for_training():
            k_states = torch.ones(ak_dist.size(), dtype=torch.long).to(ak_dist.device) * -1  # -1 means unknown
            k_states = torch.where(ak_dist < search_radius_pos, torch.ones_like(k_states), k_states)
            k_states = torch.where(ak_dist > search_radius_neg, torch.zeros_like(k_states), k_states)
        else:
            k_states = torch.zeros(ak_dist.size(), dtype=torch.long).to(ak_dist.device)
            k_states = torch.where(ak_dist < search_radius_pos, torch.ones_like(k_states), k_states)
        # relative order
        min_dist, max_dist = 2.0, search_radius_neg
        relative_orders = []
        for i in range(ak_dist.shape[0]):
            k_dist, k_state = ak_dist[i], k_states[i]
            delta_k_dist = k_dist.unsqueeze(1) - k_dist.unsqueeze(0)  # k x k
            relative_order = torch.zeros_like(delta_k_dist)
            relative_order = torch.where(delta_k_dist < -min_dist, torch.ones_like(relative_order), relative_order)
            relative_order = torch.where(delta_k_dist > min_dist, torch.ones_like(relative_order) * -1, relative_order)
            # ignore (i, j) when i/j both are far from the anchor
            ignore_index = (k_dist > max_dist).nonzero().squeeze(-1)
            mask = torch.zeros(delta_k_dist.size(), dtype=torch.int).to(ak_dist.device).bool()
            row_mask, col_mask = copy.deepcopy(mask), copy.deepcopy(mask)
            row_mask[ignore_index] = True
            col_mask[:, ignore_index] = True
            relative_order = torch.where(row_mask & col_mask, torch.zeros_like(relative_order), relative_order)
            relative_orders.append(relative_order)
        relative_orders = torch.stack(relative_orders, dim=0)  # Na x k x k
        seq_data = {
            'a_idx': np.array(valid_a_idxs, dtype=int),  # np.array
            'k_idx': k_idxs,  # np.array
            'state': k_states,  # tensor: Na x k
            'init_sim': origin_sim,  # tensor: Na x k
            'relative_order': relative_orders,  # tensor:  Na x k x k
            'a': (a_pos, a_g_desc, a_l_kpt, a_l_desc),  # tensors
            'k': (k_pos, k_g_desc, k_l_kpt, k_l_desc)  # tensors
        }
        return seq_data

    def vis_top_k(self, a_idxs, device='cpu', k=10):
        a_pts, k_pts, a_k_states = [], [], []
        seq_data = self.get_seq_top_k(a_idxs, k, device)
        if seq_data is None:
            return
        valid_a_idxs, k_idxs, origin_sim = seq_data['a_idx'], seq_data['k_idx'], seq_data['init_sim'].cpu().numpy()
        if valid_a_idxs is None:
            return
        a_pos, k_pos = seq_data['a'][0][0].cpu().numpy(), seq_data['k'][0][0].cpu().numpy()
        for i in range(len(valid_a_idxs)):
            a_pos_i = a_pos[i:i+1].repeat(k_pos.shape[0], axis=0)
            a_pts.append(a_pos_i)
            k_pts.append(k_pos)
            a_k_states.append(origin_sim[i])
        a_k_pcs = self.dataset.get_pcs(list(valid_a_idxs) + list(k_idxs), normalize=True, use_real_center=True)
        a_pts = np.concatenate(a_pts, axis=0)
        k_pts = np.concatenate(k_pts, axis=0)
        a_k_states = np.concatenate(a_k_states, axis=0)
        from utils.draw_result import draw_anchor_top_k
        draw_anchor_top_k([], a_pts, k_pts, a_k_states, title=f'anchor: {valid_a_idxs}\n, top k: {k_idxs}')


if __name__ == '__main__':
    dataset = RerankDataset('campus_test', for_training=False, pr_backbone='patch_aug_net')
    dataset.load_top_k_index(query_trip_idx=0, ref_trip_idx=1)  # query_trip_idx=0, ref_trip_idx=1
    interval, n_a, top_k = 5, 9, 25
    for i in range(0, 1000):
        a_idxs = [i + x*interval for x in range(n_a)]
        dataset.vis_top_k(a_idxs, k=top_k)
    data_loader = data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True, collate_fn=mycollate)
    for data_item in data_loader:
        for item_i in data_item:
            query_global_desc = item_i[0]
            query_local_desc = item_i[1]
            query_position = item_i[2]
            top_k_global_descs = item_i[3]
            top_k_local_descs = item_i[4]
            top_k_positions = item_i[5]
            top_k_states = item_i[6]
            print(query_position)