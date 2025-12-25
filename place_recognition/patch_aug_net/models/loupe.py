import math

import torch
from torch import nn
import torch.nn.functional as F


class MLPAttentionLayer(nn.Module):
    r""" Simple attention layer based on mlp
        Input: B x C x N
        Return: B x C x N
     """
    def __init__(self, channels=None):
        super(MLPAttentionLayer, self).__init__()
        self.mlps = nn.ModuleList()
        for i in range(len(channels)-1):
            mlp_i = nn.Conv1d(channels[i], channels[i+1], 1, bias=False)
            self.mlps.append(mlp_i)
        self.softmax = nn.Softmax(dim=-1)
        self.trans_conv = nn.Conv1d(channels[-1], channels[-1], 1)
        self.after_norm = nn.BatchNorm1d(channels[-1])
        self.act = nn.ReLU()

    def forward(self, x, return_attn=False):
        # print("x: ", x)
        x_res = x
        for mlp in self.mlps:
            x_res = mlp(x_res)
        x_res = torch.max(x_res, dim=1, keepdim=True)[0]
        x_res = x_res.squeeze(dim=1)  # B x N
        weights = self.softmax(x_res)  # B x N
        weights = weights.unsqueeze(dim=1)  # B x 1 x N
        x_res = x * weights
        #  way 1
        #x_res = self.act(self.after_norm(self.trans_conv(x - x_res)))
        #x = x + x_res  # residual link
        # way 2
        x = self.act(x + x_res)  # residual link
        if return_attn:
            return x, weights
        return x


class AdaptiveFeatureAggregator(nn.Module):
    """ B x C_in x K -> B x C_out x 1 """
    def __init__(self, C_in, K, C_out, l2_norm=True):
        """ C_in: channel of input
            K: num of 1xc features
            C_out: channel of output
        """
        super(AdaptiveFeatureAggregator, self).__init__()
        self.mlpa = MLPAttentionLayer(channels=[C_in, C_in])
        self.fc = nn.Linear(C_in * K, C_out)
        self.bn = nn.BatchNorm1d(C_out)
        self.l2_norm = l2_norm

    def forward(self, x):
        x = self.mlpa(x)
        B, C_in, K = x.size()
        x = x.view((B, C_in * K))  # B x C_in x K -> B x C_in*K
        x = self.fc(x)  # B x C_in*K -> B x C_out
        x = self.bn(x)
        if self.l2_norm:
            x = F.normalize(x)
        x = x.unsqueeze(-1)  # B x C_out -> B x C_out x 1
        return x


class GroupSALayer(nn.Module):
    r""" group self-attention layer
    Parameters
    ----------
    channels: int
        Number of channels of the input feature map
    gp: int
        Number of the divided query map in self-attention layer
    """

    def __init__(self, channels, gp):
        super().__init__()
        mid_channels = channels
        self.gp = gp
        assert mid_channels % 4 == 0
        self.q_conv = nn.Conv1d(channels, mid_channels, 1, bias=False, groups=gp)
        self.k_conv = nn.Conv1d(channels, mid_channels, 1, bias=False, groups=gp)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        r"""
        x: B x C x N
        """
        bs, ch, nums = x.size()
        x_q = self.q_conv(x)  # B x C x N
        x_q = x_q.reshape(bs, self.gp, ch // self.gp, nums)  # divide query map into 'gp' groups
        x_q = x_q.permute(0, 1, 3, 2)  # B x gp x N x C'

        x_k = self.k_conv(x)  # B x C x N
        x_k = x_k.reshape(bs, self.gp, ch // self.gp, nums)  # B x gp x C' x N

        x_v = self.v_conv(x)  # B x C x N
        energy = torch.matmul(x_q, x_k)  # B x gp x N x N
        energy = torch.sum(energy, dim=1, keepdims=False)  # B x gp x N x N -> B x N x N

        attn = self.softmax(energy)  # B x N x N -> B x N x N
        attn = attn / (1e-9 + attn.sum(dim=1, keepdims=True))  # B x N x N -> B x N x N
        x_r = torch.matmul(x_v, attn)  # B x C x N, B x N x N -> B x C x N
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r  # B x C x N + B x C x N -> B x C x N
        return x


class SALayer(nn.Module):
    r""" self-attention layer
    Parameters
    ----------
    channels: int
        Number of channels of the input feature map
    """

    def __init__(self, channels):
        super().__init__()
        mid_channels = channels
        # assert mid_channels % 4 == 0
        self.q_conv = nn.Conv1d(channels, mid_channels, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, mid_channels, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        r"""
        x: B x C x N
        """
        bs, ch, nums = x.size()
        x_q = self.q_conv(x)  # B x C x N
        x_q = x_q.permute(0, 2, 1)  # B x N x C

        x_k = self.k_conv(x)  # B x C x N

        x_v = self.v_conv(x)  # B x C x N
        energy = torch.matmul(x_q, x_k)  # B x N x N

        attn = self.softmax(energy)  # B x N x N -> B x N x N
        attn = attn / (1e-9 + attn.sum(dim=1, keepdims=True))  # B x N x N -> B x N x N
        x_r = torch.matmul(x_v, attn)  # B x C x N, B x N x N -> B x C x N
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r  # B x C x N + B x C x N -> B x C x N,  residual link
        return x


class NetVLADBase(nn.Module):
    def __init__(self, feature_size, max_samples, cluster_size, output_dim,
                 gating=True, add_batch_norm=True):
        super(NetVLADBase, self).__init__()
        self.feature_size = feature_size
        self.max_samples = max_samples
        self.output_dim = output_dim
        self.gating = gating
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size  # K
        self.softmax = nn.Softmax(dim=-1)

        self.cluster_weights = nn.Parameter(
            torch.randn(feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.cluster_weights2 = nn.Parameter(
            torch.randn(1, feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.hidden1_weights = nn.Parameter(
            torch.randn(feature_size * cluster_size, output_dim) * 1 / math.sqrt(feature_size))

        if add_batch_norm:
            self.cluster_biases = None
            self.bn1 = nn.BatchNorm1d(cluster_size)
        else:
            self.cluster_biases = nn.Parameter(
                torch.randn(cluster_size) * 1 / math.sqrt(feature_size))  # attention initialization
            self.bn1 = None

        self.bn2 = nn.BatchNorm1d(output_dim)

        if gating:
            self.context_gating = GatingContext(output_dim, add_batch_norm=add_batch_norm)

    def forward(self, x):
        x = x.transpose(1, 3).contiguous()  # B x 1024 x N x 1 -> B x 1 x N x 1024
        x = x.view((-1, self.max_samples, self.feature_size))  # B x N x 1024

        activation = torch.matmul(x, self.cluster_weights)  # B x N x 1024 X 1024 x 64 -> B x N x 64
        if self.add_batch_norm:
            # activation = activation.transpose(1,2).contiguous()
            activation = activation.view(-1, self.cluster_size)  # B x N x 64 -> BN x 64
            activation = self.bn1(activation)  # BN x 64
            activation = activation.view(-1, self.max_samples, self.cluster_size)  # BN x 64 -> B x N x 64
            # activation = activation.transpose(1,2).contiguous()
        else:
            activation = activation + self.cluster_biases  # B x N x 64 + 64 -> B x N x 64

        activation = self.softmax(activation)  # B x N x 64 --(dim=-1)--> B x N x 64

        # activation = activation[:,:,:64]
        activation = activation.view((-1, self.max_samples, self.cluster_size))  # B x N x 64

        a_sum = activation.sum(-2, keepdim=True)  # B x N x K --(dim=-2)--> B x 1 x K
        a = a_sum * self.cluster_weights2  # B x 1 x K X 1 x C x K -> B x C x K
        # element-wise multiply, broadcast mechanism

        activation = torch.transpose(activation, 2, 1)  # B x N x 64 -> B x 64 x N

        x = x.view((-1, self.max_samples, self.feature_size))  # B x N x C -> B x N x C
        vlad = torch.matmul(activation, x)  # B x K x N X B x N x C -> B x K x C
        vlad = torch.transpose(vlad, 2, 1)  # B x K x C -> B x C x K
        vlad = vlad - a  # B x C x K - B x C x K -> B x C x K

        vlad = F.normalize(vlad, dim=1, p=2).contiguous()  # B x C x K -> B x C x K
        return vlad


class SpatialPyramidNetVLAD(nn.Module):
    def __init__(self, feature_size, max_samples, cluster_size, output_dim,
                 gating=True,
                 aggregation_type=False,
                 add_batch_norm=True):
        super(SpatialPyramidNetVLAD, self).__init__()
        assert len(feature_size) == len(max_samples) == len(cluster_size) == len(output_dim)
        self.vlads = nn.ModuleList()
        for i in range(len(feature_size)):
            vlad_i = NetVLADBase(feature_size[i], max_samples[i], cluster_size[i], output_dim[i], gating, add_batch_norm)
            self.vlads.append(vlad_i)
        # hidden_weights -> MLP(feature_size[0] * sum_cluster_size, output_dim[0])
        sum_cluster_size = 0
        for i in range(len(cluster_size)):
            sum_cluster_size += cluster_size[i]
        self.gating = gating
        if self.gating:
            self.context_gating = GatingContext(output_dim[0], add_batch_norm=add_batch_norm)
        self.aggregation_type = aggregation_type
        if self.aggregation_type == 0:  # not use AdaptiveFeatureAggregator (use FC)
            self.hidden_weights = nn.Parameter(
                torch.randn(feature_size[0] * sum_cluster_size, output_dim[0]) * 1 / math.sqrt(
                    feature_size[0]))  # sum_cluster_size -> 4
            self.bn = nn.BatchNorm1d(output_dim[0])
        elif self.aggregation_type == 1:  # use AdaptiveFeatureAggregator within each scale and between scales
            self.afa_scales = nn.ModuleList()
            for i in range(len(feature_size)):
                afa_scale_i = AdaptiveFeatureAggregator(output_dim[i], cluster_size[i], output_dim[i])
                self.afa_scales.append(afa_scale_i)
            self.afa = AdaptiveFeatureAggregator(output_dim[0], len(feature_size), output_dim[0])
        elif self.aggregation_type == 2:  # use AdaptiveFeatureAggregator cross scales and regions
            self.afa = AdaptiveFeatureAggregator(output_dim[0], sum_cluster_size, output_dim[0])
        elif self.aggregation_type == 4:  # use AdaptiveFeatureAggregator within each scale
            self.afa_scales = nn.ModuleList()
            for i in range(len(feature_size)):
                afa_scale_i = AdaptiveFeatureAggregator(output_dim[i], cluster_size[i], output_dim[i])
                self.afa_scales.append(afa_scale_i)
            self.hidden_weights = nn.Parameter(
                torch.randn(feature_size[0] * len(feature_size), output_dim[0]) * 1 / math.sqrt(
                    feature_size[0]))
            self.bn = nn.BatchNorm1d(output_dim[0])
        elif self.aggregation_type == 5:  # use AdaptiveFeatureAggregator between scales
            self.hidden_weights = nn.ParameterList()
            self.bns = nn.ModuleList()
            for i in range(len(feature_size)):
                hidden_weight_i = nn.Parameter(
                torch.randn(feature_size[i] *  cluster_size[i], output_dim[i]) * 1 / math.sqrt(
                    feature_size[i]))
                self.hidden_weights.append(hidden_weight_i)
                bn_i = nn.BatchNorm1d(output_dim[i])
                self.bns.append(bn_i)
            self.afa = AdaptiveFeatureAggregator(output_dim[0], len(feature_size), output_dim[0])

    def forward(self, features=None):
        if features is None:
            features = []
        # v0: B x C x N0(=128) x 1 -> B x C x K0(=4)
        # v1: B x C x N1(=1024) x 1 -> B x C x K1(=16)
        # v2: B x C x N2(=4096) x 1 -> B x C x K2(=64)
        v_list = []
        for i in range(len(self.vlads)):
            v_i = self.vlads[i](features[i])
            v_list.append(v_i)
        # aggregate vlad features
        if self.aggregation_type == 0:
            v0123 = torch.cat(v_list, dim=-1)  # [B x C x K0, B x C x K1, B x C x K2] -> B x C x (K0+K1+K2)
            B, C, K = v0123.size()
            vlad = v0123.view((B, C * K))  # B x C x (K0+K1+K2) -> B x C*(K0+K1+K2)
            vlad = torch.matmul(vlad, self.hidden_weights)  # B x C*(K0+K1+K2) -> B x C(=256)
            vlad = self.bn(vlad)  # B x C -> B x C
            vlad = F.normalize(vlad)
        elif self.aggregation_type == 1:
            for i in range(len(v_list)):
                v_list[i] = self.afa_scales[i](v_list[i])
            v0123 = torch.cat(v_list, dim=-1)  # [B x C x 1, B x C x 1, B x C x 1] -> B x C x 3
            vlad = self.afa(v0123).squeeze(-1)
        elif self.aggregation_type == 2:
            v0123 = torch.cat(v_list, dim=-1)  # [B x C x K0, B x C x K1, B x C x K2] -> B x C x (K0+K1+K2)
            vlad = self.afa(v0123).squeeze(-1)
        elif self.aggregation_type == 3:  # not use AdaptiveFeatureAggregator (use max pooling)
            v0123 = torch.cat(v_list, dim=-1)  # [B x C x K0, B x C x K1, B x C x K2] -> B x C x (K0+K1+K2)
            vlad = F.max_pool2d(v0123, kernel_size=[1, v0123.size(2)]).squeeze(-1)  # B x C x (K0+K1+K2) -> B x C
            vlad = F.normalize(vlad)
        elif self.aggregation_type == 4:
            for i in range(len(v_list)):
                v_list[i] = self.afa_scales[i](v_list[i])
            v0123 = torch.cat(v_list, dim=-1)  # [B x C x 1, B x C x 1, B x C x 1] -> B x C x 3
            B, C, K = v0123.size()
            vlad = v0123.view((B, C * K))  # B x C x 3 -> B x C*3
            vlad = torch.matmul(vlad, self.hidden_weights)  # B x C*3 -> B x C(=256)
            vlad = self.bn(vlad)  # B x C -> B x C
            vlad = F.normalize(vlad)
        elif self.aggregation_type == 5:
            for i in range(len(v_list)):
                B, C, K = v_list[i].size()
                v_list[i] = v_list[i].view((B, C * K))
                v_list[i] = torch.matmul(v_list[i], self.hidden_weights[i])  # B x C*K -> B x C(=256)
                v_list[i] = self.bns[i](v_list[i])  # B x C -> B x C
                v_list[i] = F.normalize(v_list[i]).unsqueeze(-1)
            v0123 = torch.cat(v_list, dim=-1)  # [B x C x 1, B x C x 1, B x C x 1] -> B x C x 3
            vlad = self.afa(v0123)

        if self.gating:
            vlad = self.context_gating(vlad)  # B x C -> B x C
        return vlad  # B x 256


class GatingContext(nn.Module):
    def __init__(self, dim, add_batch_norm=True):
        super(GatingContext, self).__init__()
        self.dim = dim
        self.add_batch_norm = add_batch_norm
        self.gating_weights = nn.Parameter(
            torch.randn(dim, dim) * 1 / math.sqrt(dim))
        self.sigmoid = nn.Sigmoid()

        if add_batch_norm:
            self.gating_biases = None
            self.bn1 = nn.BatchNorm1d(dim)
        else:
            self.gating_biases = nn.Parameter(
                torch.randn(dim) * 1 / math.sqrt(dim))
            self.bn1 = None

    def forward(self, x):
        gates = torch.matmul(x, self.gating_weights)  # B x 256 X 256 x 256 -> B x 256

        if self.add_batch_norm:
            gates = self.bn1(gates)  # B x 256 -> B x 256
        else:
            gates = gates + self.gating_biases  # B x 256 + 256 -> B x 256

        gates = self.sigmoid(gates)  # B x 256 -> B x 256

        activation = x * gates  # B x 256 * B x 256 -> B x 256

        return activation


if __name__ == '__main__':
    x = torch.randn(1, 3, 5)
    for i in range(5):
        x[0, :, i] = i
    mlps = [x.shape[1], 64, 128]
    mlpa_layer = MLPAttentionLayer(mlps)
    new_x, w = mlpa_layer(x)