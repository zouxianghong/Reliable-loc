import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class OTLoss(nn.Module):
    def __init__(self):
        super(OTLoss, self).__init__()

    def forward(self, scores, pairs, unpair0, unpair1, use_unpair=True):
        ploss = 0
        uloss = 0
        nvalid = 0
        for i in range(scores.shape[0]):
            logscore = -scores[i, :, :]
            if len(pairs[i]) == 0:  # negative group has no point pairs!
                continue
            nvalid = nvalid + 1
            if len(pairs[i]) > 0:
                ploss += torch.mean(logscore[pairs[i][:, 0], pairs[i][:, 1]])
            if len(unpair0[i]) > 0 and use_unpair:
                uloss += torch.mean(logscore[unpair0[i], -1])
            if len(unpair1[i]) > 0 and use_unpair:
                uloss += torch.mean(logscore[-1, unpair1[i]])
        loss = ploss + uloss
        if nvalid > 0:
            loss = loss / nvalid
        return loss


class PPSLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(PPSLoss, self).__init__()
        self.margin = margin

    def forward(self, src_feat, tgt_feat, neg_idxs):
        """ src_feat: b x m x d, tgt_feat: b x m x d,
            pairs: list of ndarray(* x (2+num_keep)))
        """
        if tgt_feat is None:
            return 0.0
        src_feat = F.normalize(src_feat, dim=-1)
        tgt_feat = F.normalize(tgt_feat, dim=-1)
        a_vec, p_vec, n_vec = [], [], []
        for i in range(src_feat.shape[0]):
            neg_idxs_i = neg_idxs[i]
            if len(neg_idxs_i) == 0:
                continue
            a_vec_i = src_feat[i][neg_idxs_i[:, 0]]  # * x d
            p_vec_i = tgt_feat[i][neg_idxs_i[:, 1]]  # * x d
            n_vec_i = []
            for j in range(2, neg_idxs_i.shape[-1]):
                n_vec_ij = tgt_feat[i][neg_idxs_i[:, j]][:, None, :]  # * x 1 x d
                n_vec_i.append(n_vec_ij)
            n_vec_i = torch.cat(n_vec_i, dim=1)  # * x num_keep x d
            an_euc_dist = F.pairwise_distance(a_vec_i[:, None, :], n_vec_i)  # * x num_keep
            n_min_ind = torch.min(an_euc_dist, dim=-1)[1]  # *
            new_n_vec_i = []
            for j in range(n_min_ind.shape[0]):
                n_vec_ij = n_vec_i[j, n_min_ind[j], :][None, ...]  # 1 x 1 x d
                new_n_vec_i.append(n_vec_ij)
            new_n_vec_i = torch.cat(new_n_vec_i, dim=0)  # * x d
            a_vec.append(a_vec_i)
            p_vec.append(p_vec_i)
            n_vec.append(new_n_vec_i)
        if len(a_vec) == 0:
            return 0.0
        a_vec = torch.cat(a_vec, dim=0)  # * x d
        p_vec = torch.cat(p_vec, dim=0)  # * x d
        n_vec = torch.cat(n_vec, dim=0)  # * x d
        loss = 0.0
        if a_vec.shape[0] > 0:
            # a - p
            euc_dist = F.pairwise_distance(a_vec, p_vec)
            ap_loss = torch.mean(torch.pow(euc_dist, 2))
            loss += ap_loss
            # a - n
            euc_dist = F.pairwise_distance(a_vec, n_vec)
            an_loss = torch.mean(torch.pow(torch.clamp(self.margin - euc_dist, min=0.0), 2))
            loss += an_loss
        return loss