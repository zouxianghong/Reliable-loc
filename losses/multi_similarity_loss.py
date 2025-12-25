import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiSimilarityLoss(nn.Module):
    def __init__(self, thresh=0.5, margin=0.3, scale_pos=2.0, scale_neg=40.0, mode='apn'):
        super(MultiSimilarityLoss, self).__init__()
        self.mode = mode  # 'apn' or 'apn_sim' or 'ak'
        self.thresh = thresh
        self.margin = margin
        self.scale_pos = scale_pos
        self.scale_neg = scale_neg

    def forward(self, a, b, c):
        if self.mode == 'apn':
            return self._forward_apn(a, b, c)
        if self.mode == 'apn_sim':
            return self._forward_apn_sim(a, b, c)
        elif self.mode == 'ak':
            return self._forward_ak(a, b, c)
        else:
            return None

    def _forward_apn(self, a_feat, p_feat, n_feat):
        """ a/p/n: b x k x d """
        if len(a_feat.shape) == 2:
            a_feat = a_feat.unsqueeze(0)
            p_feat = p_feat.unsqueeze(0)
            n_feat = n_feat.unsqueeze(0)
        b, k, d = a_feat.size()
        loss = []
        for i in range(b):
            pos_sim = F.cosine_similarity(a_feat[i], p_feat[i])
            neg_sim = F.cosine_similarity(a_feat[i], n_feat[i])
            neg_sim = neg_sim[neg_sim + self.margin > min(pos_sim)]
            if len(pos_sim) < 1 or len(neg_sim) < 1:
                continue
            pos_loss = 1.0 / self.scale_pos * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_sim - self.thresh))))
            neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_sim - self.thresh))))
            loss.append(pos_loss + neg_loss)
        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)
        loss = torch.mean(torch.stack(loss))
        return loss

    def _forward_apn_sim(self, pos_sim, neg_sim, place_holder=None):
        """ pos_sim: b x p, neg_sim: b x n """
        loss = []
        for i in range(pos_sim.shape[0]):
            pos_sim_i = pos_sim[i]
            neg_sim_i = neg_sim[i]
            neg_sim_i = neg_sim_i[neg_sim_i + self.margin > min(pos_sim_i)]
            if len(pos_sim_i) < 1 or len(neg_sim_i) < 1:
                continue
            pos_loss = 1.0 / self.scale_pos * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_sim_i - self.thresh))))
            neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_sim_i - self.thresh))))
            loss.append(pos_loss + neg_loss)
        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)
        loss = torch.mean(torch.stack(loss))
        return loss

    def _forward_ak(self, a_feat, k_feat, k_label):
        """ a_feat: b x 1 x d, k_feat: b x k x d, k_label: b x k """
        if len(a_feat.shape) == 2:
            a_feat = a_feat.unsqueeze(0)
            k_feat = k_feat.unsqueeze(0)
            k_label = k_label.unsqueeze(0)
        b, k, d = k_feat.size()
        loss = []
        for i in range(b):
            # pos
            p_idx = (k_label[i] > 0).nonzero().squeeze(1)
            if len(p_idx) == 0:
                continue
            p_feat_i = k_feat[i][p_idx]
            a_feat_i = a_feat[i].repeat(p_feat_i.shape[0] ,1)
            pos_sim = F.cosine_similarity(a_feat_i, p_feat_i)
            # neg
            n_idx = (k_label[i] == 0).nonzero().squeeze(1)
            if len(n_idx) == 0:
                continue
            n_feat_i = k_feat[i][n_idx]
            a_feat_i = a_feat[i].repeat(n_feat_i.shape[0], 1)
            neg_sim = F.cosine_similarity(a_feat_i, n_feat_i)
            neg_sim = neg_sim[neg_sim + self.margin > min(pos_sim)]
            if len(pos_sim) < 1 or len(neg_sim) < 1:
                continue
            pos_loss = 1.0 / self.scale_pos * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_sim - self.thresh))))
            neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_sim - self.thresh))))
            loss.append(pos_loss + neg_loss)
        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)
        loss = torch.mean(torch.stack(loss))
        return loss


if __name__ == '__main__':
    # forward apn
    a_feat = torch.randn(16, 5, 256)  # b x k x d
    a_feat = F.normalize(a_feat, dim=-1)
    p_feat = torch.randn(16, 5, 256)  # b x k x d
    p_feat = F.normalize(p_feat, dim=-1)
    n_feat = torch.randn(16, 5, 256)  # b x k x d
    n_feat = F.normalize(n_feat, dim=-1)
    loss_func = MultiSimilarityLoss(mode='apn')
    loss = loss_func(a_feat, p_feat, n_feat)
    # forward ak
    a_feat = torch.randn(16, 1, 256)  # b x 1 x d
    a_feat = F.normalize(a_feat, dim=-1)
    k_feat = torch.randn(16, 5, 256)  # b x k x d
    k_feat = F.normalize(k_feat, dim=-1)
    k_label = torch.randint(0, 2, (16, 5)).float()
    loss_func = MultiSimilarityLoss(mode='ak')
    loss = loss_func(a_feat, k_feat, k_label)
    print('loss: ', loss)