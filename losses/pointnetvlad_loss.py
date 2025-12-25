import torch

import torch.nn.functional as F

from libs.emd_module.emd_module import emdModule
from libs.chamfer_dist import ChamferDistanceL1


def best_pos_distance(query, pos_vecs):
    num_pos = pos_vecs.shape[1]
    query_copies = query.repeat(1, int(num_pos), 1)
    diff = ((pos_vecs - query_copies) ** 2).sum(2)
    min_pos, _ = diff.min(1)
    max_pos, _ = diff.max(1)
    return min_pos, max_pos


def triplet_loss(q_vec, pos_vecs, neg_vecs, margin, use_min=False, lazy=False, ignore_zero_loss=False):
    min_pos, max_pos = best_pos_distance(q_vec, pos_vecs)

    # PointNetVLAD official code use min_pos, but i think max_pos should be used
    if use_min:
        positive = min_pos
    else:
        positive = max_pos

    num_neg = neg_vecs.shape[1]
    batch = q_vec.shape[0]
    query_copies = q_vec.repeat(1, int(num_neg), 1)
    positive = positive.view(-1, 1)
    positive = positive.repeat(1, int(num_neg))

    loss = margin + positive - ((neg_vecs - query_copies) ** 2).sum(2)
    loss = loss.clamp(min=0.0)
    if lazy:
        triplet_loss = loss.max(1)[0]
    else:
        triplet_loss = loss.sum(1)
    if ignore_zero_loss:
        hard_triplets = torch.gt(triplet_loss, 1e-16).float()
        num_hard_triplets = torch.sum(hard_triplets)
        triplet_loss = triplet_loss.sum() / (num_hard_triplets + 1e-16)
    else:
        triplet_loss = triplet_loss.mean()
    return triplet_loss


def triplet_loss_wrapper(q_vec, pos_vecs, neg_vecs, other_neg, m1, m2, use_min=False, lazy=False,
                         ignore_zero_loss=False):
    return triplet_loss(q_vec, pos_vecs, neg_vecs, m1, use_min, lazy, ignore_zero_loss)


def quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg, m1, m2, use_min=False, lazy=False, ignore_zero_loss=False,
                    soft_margin=False):
    min_pos, max_pos = best_pos_distance(q_vec, pos_vecs)
    # PointNetVLAD official code use min_pos, but i think max_pos should be used
    if use_min:
        positive = min_pos
    else:
        positive = max_pos

    num_neg = neg_vecs.shape[1]
    batch = q_vec.shape[0]
    query_copies = q_vec.repeat(1, int(num_neg), 1)
    positive = positive.view(-1, 1)
    positive = positive.repeat(1, int(num_neg))

    loss = m1 + positive - ((neg_vecs - query_copies) ** 2).sum(2)
    if soft_margin:
        loss = loss.clamp(max=88)
        loss = torch.log(1 + torch.exp(loss))  # soft-plus
    else:
        loss = loss.clamp(min=0.0)  # hinge function
    if lazy:  # lazy = true
        triplet_loss = loss.max(1)[0]
    else:
        triplet_loss = loss.mean(1)
    if ignore_zero_loss:  # false
        hard_triplets = torch.gt(triplet_loss, 1e-16).float()
        num_hard_triplets = torch.sum(hard_triplets)
        triplet_loss = triplet_loss.sum() / (num_hard_triplets + 1e-16)
    else:
        triplet_loss = triplet_loss.mean()

    other_neg_copies = other_neg.repeat(1, int(num_neg), 1)
    second_loss = m2 + positive - ((neg_vecs - other_neg_copies) ** 2).sum(2)
    if soft_margin:
        second_loss = second_loss.clamp(max=88)
        second_loss = torch.log(1 + torch.exp(second_loss))
    else:
        second_loss = second_loss.clamp(min=0.0)
    if lazy:
        second_loss = second_loss.max(1)[0]
    else:
        second_loss = second_loss.mean(1)
    if ignore_zero_loss:
        hard_second = torch.gt(second_loss, 1e-16).float()
        num_hard_second = torch.sum(hard_second)
        second_loss = second_loss.sum() / (num_hard_second + 1e-16)
    else:
        second_loss = second_loss.mean()

    total_loss = triplet_loss + second_loss

    return total_loss


def contrastive_quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg, m1, m2, use_min=False, lazy=True,
                                ignore_zero_loss=False, soft_margin=False):
    min_pos, max_pos = best_pos_distance(q_vec, pos_vecs)

    # PointNetVLAD official code use min_pos, but i think max_pos should be used
    if use_min:
        positive = min_pos
    else:
        positive = max_pos

    num_neg = neg_vecs.shape[1]
    batch = q_vec.shape[0]
    query_copies = q_vec.repeat(1, int(num_neg), 1)

    negative = ((neg_vecs - query_copies) ** 2).sum(2)  # [B, num_neg]
    min_neg = negative.min(1)[0]
    mask = min_neg < positive
    loss1 = loss2 = 0
    if mask.sum() != 0:
        loss1 = m1 + positive[mask].detach() - min_neg[mask]
        loss1 = loss1.clamp(min=0.0).sum()
    mask = ~mask
    if mask.sum() != 0:
        loss2 = m1 + positive[mask] - min_neg[mask]
        loss2 = loss2.clamp(min=0.0).sum()
    triplet_loss = (loss1 + loss2) / batch

    other_neg_copies = other_neg.repeat(1, int(num_neg), 1)
    positive = positive.view(-1, 1)
    positive = positive.repeat(1, int(num_neg))
    second_loss = m2 + positive - ((neg_vecs - other_neg_copies) ** 2).sum(2)
    second_loss = second_loss.clamp(min=0.0)

    if lazy:
        second_loss = second_loss.max(1)[0]
    else:
        second_loss = second_loss.mean(1)
    if ignore_zero_loss:
        hard_second = torch.gt(second_loss, 1e-16).float()
        num_hard_second = torch.sum(hard_second)
        second_loss = second_loss.sum() / (num_hard_second + 1e-16)
    else:
        second_loss = second_loss.mean()

    total_loss = triplet_loss + second_loss

    return total_loss


def hphn_quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg, m1, m2, use_min=False, lazy=False,
                         ignore_zero_loss=False):
    min_pos, max_pos = best_pos_distance(q_vec, pos_vecs)  # [B]
    min_neg, max_neg = best_pos_distance(q_vec, neg_vecs)  # [B]
    min_other_neg, max_other_neg = best_pos_distance(other_neg, neg_vecs)  # [B]

    hard_neg = torch.stack([min_neg, min_other_neg], dim=1).min(dim=1, keepdim=False)[0]
    hphn_quadruplet = m1 + max_pos - hard_neg
    hphn_quadruplet = hphn_quadruplet.clamp(min=0.0)

    return hphn_quadruplet.mean()


def contrastive_loss(q_vec, pos_vec, neg_vec, margin):
    total_loss = 0.0
    # query_copies = F.normalize(torch.stack(q_vec, dim=0))
    query_copies = torch.stack(q_vec, dim=0)
    if len(pos_vec) > 0:
        # pos_copies = F.normalize(torch.stack(pos_vec, dim=0))
        pos_copies = torch.stack(pos_vec, dim=0)
        euc_dist = F.pairwise_distance(query_copies, pos_copies)
        qp_loss = torch.mean(torch.pow(euc_dist, 2))
        total_loss += qp_loss
    if len(neg_vec) > 0:
        # neg_copies = F.normalize(torch.stack(neg_vec, dim=0))
        neg_copies = torch.stack(neg_vec, dim=0)
        euc_dist = F.pairwise_distance(query_copies, neg_copies)
        qn_loss = torch.mean(torch.pow(torch.clamp(margin - euc_dist, min=0.0), 2))
        total_loss += qn_loss
    return total_loss


def chamfer_loss(pc1, pc2):
    # pc1
    pcs1_tensor_list = []
    for i in range(len(pc1)):
        pcs1_tensor_list.append(pc1[i].float())
    feed_tensor = torch.cat(pcs1_tensor_list, 1).squeeze(0)
    # pc2
    pcs2_tensor_list = []
    for i in range(len(pc2)):
        pcs2_tensor_list.append(pc2[i].float())
    res_tensor = torch.cat(pcs2_tensor_list, 1).squeeze(0)
    loss_func = ChamferDistanceL1()
    loss = loss_func(feed_tensor, res_tensor)
    return loss


def emd_loss(pc1, pc2):
    # pc1
    pcs1_tensor_list = []
    for i in range(len(pc1)):
        pcs1_tensor_list.append(pc1[i].float())
    feed_tensor = torch.cat(pcs1_tensor_list, 1)
    feed_tensor = feed_tensor.view((-1, 4096, 3))
    # pc2
    pcs2_tensor_list = []
    for i in range(len(pc2)):
        pcs2_tensor_list.append(pc2[i].float())
    res_tensor = torch.cat(pcs2_tensor_list, 1)
    res_tensor = res_tensor.view((-1, 4096, 3))
    emd = emdModule()
    dis, _ = emd(feed_tensor, res_tensor, 0.02, 1024)
    dis = torch.mean(torch.sqrt(dis), dim=1)
    return torch.mean(dis)


def point_pair_loss(pc1, pc2):  # similar to MSE loss
    # pc1
    pcs1_tensor_list = []
    for i in range(len(pc1)):
        pcs1_tensor_list.append(pc1[i].float())
    feed_tensor = torch.cat(pcs1_tensor_list, 1)
    feed_tensor = feed_tensor.view((-1, 4096, 3))
    # pc2
    pcs2_tensor_list = []
    for i in range(len(pc2)):
        pcs2_tensor_list.append(pc2[i].float())
    res_tensor = torch.cat(pcs2_tensor_list, 1)
    res_tensor = res_tensor.view((-1, 4096, 3))
    pdist = torch.nn.PairwiseDistance(p=2)
    dists = pdist(feed_tensor, res_tensor)
    return torch.mean(dists)


def patch_chamfer_loss(origin_patches, recon_patches):
    feed_tensor = torch.cat(origin_patches, 0)
    res_tensor = torch.cat(recon_patches, 0)
    loss_func = ChamferDistanceL1()
    loss = loss_func(feed_tensor, res_tensor)
    return loss


def patch_emd_loss(origin_patches, recon_patches):
    feed_tensor = torch.cat(origin_patches, 0)
    res_tensor = torch.cat(recon_patches, 0)
    emd = emdModule()
    dis, _ = emd(feed_tensor, res_tensor, 0.02, 1024)
    dis = torch.mean(torch.sqrt(dis), dim=1)
    return torch.mean(dis)
