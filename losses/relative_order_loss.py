import torch
from torch import nn


class RelativeOrderLoss(nn.Module):
    """  """
    def __init__(self, type='roc', alpha=0.1):
        super(RelativeOrderLoss, self).__init__()
        self.alpha = alpha
        self.type = type

    def ROCLoss(self, pred_delta_dist, gt_order_mat):
        """ pred_dist: b x m x n; gt_order_mat: b x m x n """
        sigma = torch.where(gt_order_mat > 0, gt_order_mat, torch.zeros_like(gt_order_mat))
        penalty = torch.exp(self.alpha * pred_delta_dist)
        loss = torch.sum(torch.mul(sigma, penalty))
        return loss

    def MOCLoss(self, pred_delta_dist, gt_order_mat):
        """ pred_dist: b x m x n; gt_order_mat: b x m x n """
        sigma = torch.where(pred_delta_dist < 0, pred_delta_dist * -1, torch.zeros(pred_delta_dist))
        penalty = torch.mul((1 - gt_order_mat), torch.log(1 - pred_delta_dist) / torch.log(torch.Tensor(self.alpha)))
        loss = torch.sum(torch.mul(sigma, penalty))
        return loss

    def OLoss(self, pred_order_mat, gt_order_mat):
        """ pred_order_mat: b x m x n; gt_order_mat: b x m x n """
        delta = pred_order_mat - gt_order_mat
        loss = torch.sum(delta ** 2)
        return loss

    def forward(self, pred_delta_dist, gt_order_mat, pred_order_mat=None):
        b, m, n = gt_order_mat.size()
        loss = 0
        if self.type == 'roc':
            loss = loss + self.ROCLoss(pred_delta_dist, gt_order_mat)
        else:
            loss = loss + self.MOCLoss(pred_delta_dist, gt_order_mat)
        if pred_order_mat is not None:
            loss = loss + self.OLoss(pred_order_mat, gt_order_mat)
        loss = loss / (b * m * n)
        return loss