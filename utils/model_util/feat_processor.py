import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from utils.model_util.transformer import TransformerEncoderLayer
from place_recognition.patch_aug_net.models.loupe import NetVLADBase
from utils.model_util.pool import get_pool
from utils.train_util import nn_dist, nn_angle


def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                # layers.append(nn.BatchNorm1d(channels[i]))
                layers.append(nn.InstanceNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([copy.deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, prob = attention(query, key, value)
        # self.prob.append(prob)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names
        self.only_self_attn = True
        for name in layer_names:
            if name == 'cross':
                self.only_self_attn = False
                break

    def forward(self, desc0, desc1=None):
        """ desc0: b x m x d, desc1: b x n x d """
        # only self-attn
        if self.only_self_attn or desc1 is None:
            desc0 = desc0.permute(0, 2, 1)  # b x d x m
            for layer, name in zip(self.layers, self.names):
                delta0 = layer(desc0, desc0)
                desc0 = desc0 + delta0
            desc0 = desc0.permute(0, 2, 1)  # b x m x d
            return desc0
        # with cross-attn
        desc0 = desc0.permute(0, 2, 1)  # b x d x m
        desc1 = desc1.permute(0, 2, 1)  # b x d x n
        for layer, name in zip(self.layers, self.names):
            layer.attn.prob = []
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        desc0 = desc0.permute(0, 2, 1)  # b x m x d
        desc1 = desc1.permute(0, 2, 1)  # b x n x d
        return desc0, desc1


class AbsCoordEncoder(nn.Module):
    """ Input: B x N x 2 or B x N x 3
        Returns: B x N x d
    """
    def __init__(self, coord_dim, embed_dim):
        super(AbsCoordEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(coord_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.fc(x)


class DistanceEncoder(nn.Module):
    """ Input: B x N x 2 or B x N x 3
        Returns: B x N x d
    """
    def __init__(self, in_dim, embed_dim, max_dist=None):
        super(DistanceEncoder, self).__init__()
        self.max_dist = max_dist
        self.fc = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        dist = nn_dist(x).float()  # B x N x N
        if self.max_dist is not None:
            max_dist_fill = torch.ones_like(dist) * self.max_dist
            dist = torch.where(dist > self.max_dist, max_dist_fill, dist)
        x = self.fc(dist / torch.max(dist))  # B x N x d
        return x


class AngleEncoder(nn.Module):
    """ Input: B x N x 2 or B x N x 3
        Returns: B x N x d
    """
    def __init__(self, N, embed_dim, angle_k=None):
        super(AngleEncoder, self).__init__()
        self.angle_k = angle_k
        self.fc = nn.Sequential(
            nn.Linear(N, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )
        self.max_pool = nn.AdaptiveMaxPool1d(1)
    
    def forward(self, x):
        x = F.normalize(nn_angle(x, self.angle_k), dim=-1)  # b x k x m x m
        x = self.fc(x)  # b x k x m x d
        x = x.permute(0, 2, 3, 1).contiguous()  # b x m x d x k
        b, m, d, k = x.size()
        x = self.max_pool(x.view(b * m, d, k))  # bm x d x 1
        x = x.squeeze(-1).view(b, m, d)  # b x m x d
        return x


class GeoEncoder(nn.Module):
    """ Geometric structure encoder """
    def __init__(self, encode_type, num_element, element_dim, angle_k=3):
        super(GeoEncoder, self).__init__()
        self.geo_encode_type = encode_type
        if self.geo_encode_type == 'abs_coord':
            self.encoder = AbsCoordEncoder(3, element_dim)
        elif self.geo_encode_type == 'nn_dist':
            self.encoder = DistanceEncoder(num_element, element_dim)
        elif self.geo_encode_type == 'nn_angle':
            self.encoder = AngleEncoder(num_element, element_dim, angle_k)
        else:  # nn dist and nn angle
            self.geo_encoder = DistanceEncoder(num_element, element_dim)
            self.angle_encoder = AngleEncoder(num_element, element_dim, angle_k)

    def forward(self, x):
        """ x: b x m x 3 """
        if self.geo_encode_type == 'abs_coord' or self.geo_encode_type == 'nn_dist' or self.geo_encode_type == 'nn_angle':
            x = self.encoder(x)  # b x m x d
        else:
            x = self.geo_encoder(x) + self.angle_encoder(x)
        return x


class SingleFeatProcessor(nn.Module):
    """ x_global: b x d, x_local: b x n x d, x_position: b x n x 3
        out: b x out_dim, see def of out_dim
    """
    def __init__(self, config):
        super(SingleFeatProcessor, self).__init__()
        assert (config['embed_dim'] % 2 == 0)
        # encode key points' geometric info into local feature
        self.use_geo_encoder = config['use_geo_encoder']
        if self.use_geo_encoder:
            self.geo_encoder = GeoEncoder(config['geo_encode_type'], config['num_local'], config['local_dim'])

        # how to use global feature: add / cat / none
        self.add_or_cat = config['add_or_cat']  # add / cat global and local feat, or do not use global feature
        g_dim = config['global_dim'] if self.add_or_cat == 'cat' else 0
        self.mlp = nn.Sequential(
            nn.Linear(g_dim + config['local_dim'], config['embed_dim']),
            nn.LayerNorm(config['embed_dim']),
            nn.ReLU()
        )

        # aggregate local feature to reduce computation cost. do not use it in pose estimation network
        self.use_vlad = config['use_vlad']
        if self.use_vlad:
            self.vlad = NetVLADBase(feature_size=config['embed_dim'], max_samples=config['num_local'],
                                    cluster_size=config['cluster_size'], output_dim=config['embed_dim'])

        # feature interaction
        self.use_transformer = len(config['layer_names']) > 0
        if self.use_transformer:
            self.norm1 = nn.LayerNorm(config['embed_dim'])
            self.gnn_layer = AttentionalGNN(config['embed_dim'], config['layer_names'])
        # pooling
        self.use_pool = config['use_pool']
        if self.use_pool:
            self.pool = get_pool(config['pool'])
        # output dim
        self.out_dim = config['embed_dim']
        if not self.use_pool:
            d = config['embed_dim']
            n = config['cluster_size'] if self.use_vlad else config['num_local']
            self.out_dim = n * d

    def forward(self, x_global, x_local, x_position):
        """ x_global: b x d, x_local: b x n x d, x_position: b x n x 3
            out: b x out_dim, see def of out_dim
        """
        # encode key points' geometric info into local feature
        if self.use_geo_encoder:
            x_local = x_local + self.geo_encoder(x_position)  # b x n x d

        # use global feature
        if self.add_or_cat == 'add':
            x_feat = x_global + x_local  # b x n x d
        elif self.add_or_cat == 'cat':
            x_global = x_global.unsqueeze(1).repeat(1, x_local.shape[1], 1)  # b x n x d
            x_feat = torch.cat([x_global, x_local], dim=-1)  # b x n x 2d
        else:
            x_feat = x_local  # b x n x d
        x_feat = self.mlp(x_feat)  # b x n x d

        # aggregate local feature to reduce computation cost. do not use it in pose estimation network
        if self.use_vlad:
            x_feat = self.vlad(x_feat.permute(0, 2, 1).unsqueeze(3)).permute(0, 2, 1)  # b x n x d

        # feature interaction
        if self.use_transformer:
            x_feat = self.norm1(x_feat)  # b x n x d
            x_feat = self.gnn_layer(x_feat, x_feat)  # b x n x d
        # pool
        if self.use_pool:
            x_feat = x_feat.permute(0, 2, 1)  # b x d x n
            x_feat = rearrange(x_feat, 'b d n -> b d n 1')  # b x d x n x 1
            x_feat = self.pool(x_feat)  # b x d x 1 x 1
        return x_feat  # b x d x 1 x 1 or b x n x d


class PairwiseFeatProcessor(nn.Module):
    """ x_global: b x d, x_local: b x m x d, x_position: b x m x 3
        y_global: b x d, y_local: b x n x d, y_position: b x n x 3
        out: [b x out_dim, b x out_dim], see def of out_dim
    """
    def __init__(self, config):
        super(PairwiseFeatProcessor, self).__init__()
        assert (config['embed_dim'] % 2 == 0)
        # encode key points' geometric info into local feature
        self.use_geo_encoder = config['use_geo_encoder']
        if self.use_geo_encoder:
            self.geo_encoder = GeoEncoder(config['geo_encode_type'], config['num_local'], config['local_dim'])

        # how to use global feature: add / cat / none
        self.add_or_cat = config['add_or_cat']  # add / cat global and local feat, or do not use global feature
        g_dim = config['global_dim'] if self.add_or_cat == 'cat' else 0
        self.mlp = nn.Sequential(
            nn.Linear(g_dim + config['local_dim'], config['embed_dim']),
            nn.LayerNorm(config['embed_dim']),
            nn.ReLU()
        )

        # aggregate local feature to reduce computation cost. do not use it in pose estimation network
        self.use_vlad = config['use_vlad']
        if self.use_vlad:
            self.vlad = NetVLADBase(feature_size=config['embed_dim'], max_samples=config['num_local'],
                                    cluster_size=config['cluster_size'], output_dim=config['embed_dim'])

        # feature interaction
        self.use_transformer = len(config['layer_names']) > 0
        if self.use_transformer:
            self.norm1 = nn.LayerNorm(config['embed_dim'])
            self.gnn_layer = AttentionalGNN(config['embed_dim'], config['layer_names'])
        # pooling
        self.use_pool = config['use_pool']
        if self.use_pool:
            self.pool = get_pool(config['pool'])
        # output dim
        self.out_dim = config['embed_dim']
        if not self.use_pool:
            d = config['embed_dim']
            n = config['cluster_size'] if self.use_vlad else config['num_local']
            self.out_dim = n * d

    def forward(self, x_global, x_local, x_position, y_global, y_local, y_position):
        """ x_global: b x d, x_local: b x m x d, x_position: b x m x 3
            y_global: b x d, y_local: b x n x d, y_position: b x n x 3
        """
        # encode key points' geometric info into local feature
        if self.use_geo_encoder:
            x_local = x_local + self.geo_encoder(x_position)  # b x m x d
            y_local = y_local + self.geo_encoder(y_position)  # b x n x d

        # add or cat
        if self.add_or_cat == 'add':
            x_feat = x_global + x_local  # b x m x d
            y_feat = y_global + y_local  # b x n x d
        elif self.add_or_cat == 'cat':
            x_global = x_global.unsqueeze(1).repeat(1, x_local.shape[1], 1)  # b x m x d
            y_global = y_global.unsqueeze(1).repeat(1, y_local.shape[1], 1)  # b x n x d
            x_feat = torch.cat([x_global, x_local], dim=-1)  # b x m x 2d
            y_feat = torch.cat([y_global, y_local], dim=-1)  # b x n x 2d
        else:
            x_feat = x_local  # b x m x d
            y_feat = y_local  # b x n x d
        x_feat = self.mlp(x_feat)  # b x m x d
        y_feat = self.mlp(y_feat)  # b x n x d

        # aggregate local feature to reduce computation cost. do not use it in pose estimation network
        if self.use_vlad:
            x_feat = self.vlad(x_feat.permute(0, 2, 1).unsqueeze(3)).permute(0, 2, 1)  # b x cluster x d
            y_feat = self.vlad(y_feat.permute(0, 2, 1).unsqueeze(3)).permute(0, 2, 1)  # b x cluster x d

        if self.use_transformer:
            x_feat = self.norm1(x_feat)  # b x m x d
            y_feat = self.norm1(y_feat)  # b x n x d
            x_feat, y_feat = self.gnn_layer(x_feat, y_feat)  # b x m x d, b x n x d
        # pooling
        if self.use_pool:
            x_feat = x_feat.permute(0, 2, 1)  # b x d x m
            y_feat = y_feat.permute(0, 2, 1)  # b x d x n
            x_feat = rearrange(x_feat, 'b d m -> b d m 1')  # b x d x m x 1
            y_feat = rearrange(y_feat, 'b d n -> b d n 1')  # b x d x n x 1
            x_feat = self.pool(x_feat)  # b x d x 1 x 1
            y_feat = self.pool(y_feat)  # b x d x 1 x 1
        return x_feat, y_feat  # b x d x 1 x 1 or b x n x d
