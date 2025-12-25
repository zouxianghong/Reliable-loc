#################### Refer to Spectral Geometric Verification ####################

# Functions in this file are adapted from: https://github.com/ZhiChen902/SC2-PCR/blob/main/SC2_PCR.py

import numpy as np
import torch

def match_pair_parallel(src_keypts, tgt_keypts, src_features, tgt_features, mask=None):
    ''' mask: b x n_src x n_tgt, 0: potenial matches, 1: masked matches
    '''
    distance = torch.cdist(src_features, tgt_features)
    if mask is not None:
        max_dist = torch.max(distance) * mask
        distance = torch.where(mask == 0, distance, max_dist)
    min_vals, min_ids = torch.min(distance, dim=2)
 
    kpt_min_ids = min_ids.unsqueeze(-1).expand(-1, -1, tgt_keypts.shape[-1])
    tgt_keypts_corr = torch.gather(tgt_keypts, 1, kpt_min_ids)
    src_keypts_corr = src_keypts
    
    feat_min_ids = min_ids.unsqueeze(-1).expand(-1, -1, tgt_features.shape[-1])
    src_features_corr = src_features
    tgt_features_corr = torch.gather(tgt_features, 1, feat_min_ids)

    return src_keypts_corr, tgt_keypts_corr, src_features_corr, tgt_features_corr

def power_iteration(M, num_iterations=5):
    """
    Calculate the leading eigenvector using power iteration algorithm
    Input:
        - M:      [bs, num_pts, num_pts] the adjacency matrix
    Output:
        - leading_eig: [bs, num_pts] leading eigenvector
    """
    leading_eig = torch.ones_like(M[:, :, 0:1])
    leading_eig_last = leading_eig
    for i in range(num_iterations):
        leading_eig = torch.bmm(M, leading_eig)
        leading_eig = leading_eig / (torch.norm(leading_eig, dim=1, keepdim=True) + 1e-6)
        if torch.allclose(leading_eig, leading_eig_last):
            break
        leading_eig_last = leading_eig
    leading_eig = leading_eig.squeeze(-1)
    return leading_eig


def cal_spatial_consistency( M, leading_eig):
    """
    Calculate the spatial consistency based on spectral analysis.
    Input:
        - M:          [bs, num_pts, num_pts] the adjacency matrix
        - leading_eig [bs, num_pts]           the leading eigenvector of matrix M
    Output:
        - sc_score_list [bs, 1]
    """
    spatial_consistency = leading_eig[:, None, :] @ M @ leading_eig[:, :, None]
    spatial_consistency = spatial_consistency.squeeze(-1) / M.shape[1]
    return spatial_consistency


def sgv_parallel(src_keypts, tgt_keypts, src_features, tgt_features, mask=None, d_thresh=5.0, k=32):
    """
    Input:
        - src_keypts: [1, num_pts, 3]
        - tgt_keypts: [bs, num_pts, 3]
        - src_features: [1, num_pts, D]
        - tgt_features: [bs, num_pts, D]
    Output:
        - sc_score_list:   [bs, 1], spatial consistency score for each candidate
    """
    # normalize:
    src_features = torch.nn.functional.normalize(src_features, p=2.0, dim=1)
    tgt_features = torch.nn.functional.normalize(tgt_features, p=2.0, dim=1)
    
    # Correspondence Estimation: Nearest Neighbour Matching
    src_keypts_corr, tgt_keypts_corr, src_feats_corr, tgt_feats_corr = match_pair_parallel(src_keypts, tgt_keypts, src_features, tgt_features, mask=mask)

    # Spatial Consistency Adjacency Matrix
    src_dist = torch.norm((src_keypts_corr[:, :, None, :] - src_keypts_corr[:, None, :, :]), dim=-1)
    target_dist = torch.norm((tgt_keypts_corr[:, :, None, :] - tgt_keypts_corr[:, None, :, :]), dim=-1)
    cross_dist = torch.abs(src_dist - target_dist)
    adj_mat = torch.clamp(1.0 - cross_dist ** 2 / d_thresh ** 2, min=0)

    # Spatial Consistency Score
    lead_eigvec = power_iteration(adj_mat)
    sc_score_list = cal_spatial_consistency(adj_mat, lead_eigvec)
    
    # top k key point correspondences
    _, top_k_indices = torch.topk(lead_eigvec, k=k, dim=-1)
    b = top_k_indices.shape[0]
    top_k_src_kpts, top_k_tgt_kpts, top_k_weights = [], [], []
    for i in range(b):
        src_kpts_i = src_keypts_corr[i].index_select(index=top_k_indices[i], dim=0)
        tgt_kpts_i = tgt_keypts_corr[i].index_select(index=top_k_indices[i], dim=0)
        src_feats_i = src_feats_corr[i].index_select(index=top_k_indices[i], dim=0)  # k x d
        tgt_feats_i = tgt_feats_corr[i].index_select(index=top_k_indices[i], dim=0)  # k x d
        weights_i = torch.cosine_similarity(src_feats_i, tgt_feats_i, dim=-1)
        top_k_src_kpts.append(src_kpts_i)
        top_k_tgt_kpts.append(tgt_kpts_i)
        top_k_weights.append(weights_i)
    top_k_src_kpts = torch.stack(top_k_src_kpts, dim=0)
    top_k_tgt_kpts = torch.stack(top_k_tgt_kpts, dim=0)
    top_k_weights = torch.stack(top_k_weights, dim=0)

    sc_score_list = np.squeeze(sc_score_list.cpu().detach().numpy())
    return sc_score_list, top_k_src_kpts, top_k_tgt_kpts, top_k_weights


def sgv_fn(src_lfs, src_kps, tgt_lfs, tgt_kps, mask=None, d_thresh=5.0, k=32, use_cuda=True):
    if use_cuda:
        src_kps = src_kps.cuda()
        src_lfs = src_lfs.cuda()
        tgt_kps = tgt_kps.cuda()
        tgt_lfs = tgt_lfs.cuda()

    conf_list, src_kps, tgt_kps, weights = sgv_parallel(src_kps, tgt_kps, src_lfs, tgt_lfs, mask=mask, d_thresh=d_thresh, k=k)

    return  conf_list, src_kps.cpu().detach().numpy(), tgt_kps.cpu().detach().numpy(), weights.cpu().detach().numpy()
