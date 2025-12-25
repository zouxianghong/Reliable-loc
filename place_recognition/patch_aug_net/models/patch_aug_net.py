import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from typing import List, Tuple, Optional, Any

import loupe as lp
from libs.pointops.functions import pointops
from utils.model_util import pt_util
from pointnet_autoencoder import PointNetDecoder

__all__ = ['Network']


class Network(nn.Module):
    def __init__(self, param=None, use_a2a_recon=False, use_l2_norm=False):
        super(Network, self).__init__()
        # backbone
        self.backbone = PointNet2(param=param)

        # task1: global descriptor
        aggregation = param["AGGREGATION"]
        if aggregation == 'spvlad':
            self.aggregation = lp.SpatialPyramidNetVLAD(
                feature_size=param["FEATURE_SIZE"],  # 256,256,256,256
                max_samples=param["MAX_SAMPLES"],  # 64,256,1024,4096
                cluster_size=param["CLUSTER_SIZE"],  # 1,4,16,64
                output_dim=param["OUTPUT_DIM"],  # 256,256,256,256
                gating=param['GATING'],  # True
                aggregation_type=param['AGGREGATION_TYPE'],  # 1~5
                add_batch_norm=True)
        else:
            print("No aggregation algorithm: ", aggregation)

        # task2: patch reconstruction
        self.use_l2_norm = use_l2_norm
        self.use_a2a_recon = use_a2a_recon
        if self.use_a2a_recon:
            self.decoder = PointNetDecoder(embedding_size=256, num_points=param["KNN"][0])

    def forward(self, x, nn_dict=None, return_feat=True):
        r"""
        x: B x 1 x N x 3
        """
        # task1: global descriptor
        x = x.squeeze(1)
        xyz = x  # B x N x 3
        # center_idx: [B x 1024, B x 128, B x 16]
        # sample_idx: [B x 1024 x nsample, B x 128 x nsample, B x 16 x nsample]
        # sa_features: [B x 64 x 1024, B x 256 x 128, B x 512 x 16]
        # fp_features: [B x 256 x 128, B x 256 x 1024, B x 256 x 4096]
        res = self.backbone(x)
        center_idx = res['center_idx_origin']
        sample_idx = res['sample_idx_origin']
        #sa_features = res['sa_features']
        fp_features = res['fp_features']

        x = self.aggregation(fp_features)  # Bx256x128, Bx256x1024, Bx256x4096 -> Bx256
        res = x
        # task2: patch reconstruction and patch feature augmentation (f1 only, each cloud has 1024 patches)
        if nn_dict is not None:  # nn_dict: key: indices of two clouds, value: list of nearest idx pairs
            # get related clouds' indices
            related_cloud_idx = set()
            for i, j in nn_dict:
                related_cloud_idx.add(i)
                related_cloud_idx.add(j)
            related_cloud_idx = list(related_cloud_idx)
            # get origin and reconstructed patches
            center_indices = []
            origin_patches_out = []
            patch_features = []
            reconstructed_patches_out = []
            origin_patches = pointops.grouping(
                xyz.transpose(1, 2).contiguous(),
                sample_idx[0])  # B x 3 x 1024 x nsample
            for i in range(len(related_cloud_idx)):
                cloud_idx = torch.tensor([related_cloud_idx[i]]).to(x.device)
                center_indices_i = torch.index_select(center_idx[0], dim=0, index=cloud_idx)  # 1024
                selected_features_i = torch.index_select(fp_features[1], dim=0, index=cloud_idx)
                selected_features_i = selected_features_i.squeeze().transpose(1, 0)  # 1024 x 256
                if self.use_l2_norm:
                    selected_features_i = F.normalize(selected_features_i)
                origin_patches_i = torch.index_select(origin_patches, dim=0, index=cloud_idx)
                origin_patches_i = origin_patches_i.squeeze().transpose(2, 0).transpose(1, 0)
                center_indices.append(center_indices_i)
                origin_patches_out.append(origin_patches_i)
                patch_features.append(selected_features_i)
                # a2a recon
                if self.use_a2a_recon:
                    reconstructed_patches_i = self.decoder(selected_features_i)  # 1024 x nsample x 3
                    reconstructed_patches_out.append(reconstructed_patches_i)
            patch_recon_data = {'cloud_indices': related_cloud_idx,
                                'center_indices': center_indices,
                                'origin_patches': origin_patches_out,
                                'patch_features': patch_features,
                                'reconstructed_patches': reconstructed_patches_out}
            res = res, patch_recon_data
        if return_feat:
            res = res, fp_features, center_idx
        return res


class PointNet2(nn.Module):
    r""" Modified PointNet++ (use EdgeConv and group self-attention module)
    """

    def __init__(self, param=None):
        super().__init__()
        c = 3
        sap = param['SAMPLING']
        knn = param['KNN']
        knn_dilation = param['KNN_DILATION']
        gp = param['GROUP']
        use_xyz = True
        self.use_origin_pc_in_fp = param['USE_ORIGIN_PC_IN_FP']
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointNet2SAModule(npoint=sap[0], nsample=knn[0], knn_dilation=knn_dilation, gp=gp, mlp=[c, 32, 32, 64],
                              use_xyz=use_xyz))
        self.SA_modules.append(
            PointNet2SAModule(npoint=sap[1], nsample=knn[1], knn_dilation=knn_dilation, gp=gp, mlp=[64, 64, 64, 256],
                              use_xyz=use_xyz))
        self.SA_modules.append(
            PointNet2SAModule(npoint=sap[2], nsample=knn[2], knn_dilation=knn_dilation, gp=gp, mlp=[256, 256, 256, 512],
                              use_xyz=use_xyz))
        fs = param['FEATURE_SIZE']
        self.FP_modules = nn.ModuleList()
        if not self.use_origin_pc_in_fp:
            c = 0
        self.FP_modules.append(PointNet2FPModule(mlp=[fs[1] + c, 256, 256, fs[0]]))
        self.FP_modules.append(PointNet2FPModule(mlp=[fs[2] + 64, 256, fs[1]]))
        self.FP_modules.append(PointNet2FPModule(mlp=[512 + 256, 256, fs[2]]))

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        r"""
            Forward pass of the network
            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        l_xyz = [pointcloud]  # l_xyz[0]: B x N x 3
        l_features = [pointcloud.transpose(1, 2).contiguous()]  # l_features[0]: B x 3 x N
        l_center_idx = []
        l_sample_idx = []
        # set abstraction
        # l0_xyz: B x N(=4096) x 3, l0_feat: B x C(=3) x N(=4096)
        # l1_xyz: B x N(=1024) x 3, l1_feat: B x C(=64) x N(=1024)
        # l2_xyz: B x N(=128) x 3, l2_feat: B x C(=256) x N(=128)
        # l3_xyz: B x N(=16) x 3, l3_feat: B x C(=512) x N(=16)
        for i in range(len(self.SA_modules)):
            li_xyz, li_center_idx, li_sample_idx, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
            l_center_idx.append(li_center_idx)
            l_sample_idx.append(li_sample_idx)
        sa_features = l_features
        # get center idx and sample idx in origin cloud
        l_center_idx_origin = [l_center_idx[0]]
        l_sample_idx_origin = [l_sample_idx[0]]
        for i in range(1, len(l_center_idx)):
            li_center_idx_origin = torch.gather(l_center_idx_origin[i - 1], -1, l_center_idx[i].long())
            temp_l_center_idx_origin = l_center_idx_origin[i - 1].unsqueeze(1)
            temp_l_center_idx_origin = temp_l_center_idx_origin.repeat(1, l_sample_idx[i].shape[1], 1)
            li_sample_idx_origin = torch.gather(temp_l_center_idx_origin, -1, l_sample_idx[i].long())
            l_center_idx_origin.append(li_center_idx_origin)
            l_sample_idx_origin.append(li_sample_idx_origin)
        # feature up sampling and fusion
        # l3: mlp(cat(up_sample(l4_xyz) + l3_xyz)), B x C(=256) x N(=16)
        # l2: mlp(cat(up_sample(l3_xyz) + l2_xyz)), B x C(=256) x N(=128)
        # l1: mlp(cat(up_sample(l2_xyz) + l1_xyz)), B x C(=256) x N(=1024)
        # l0: mlp(cat(up_sample(l1_xyz) + l0_xyz)), B x C(=256) x N(=4096)
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            if i == -len(self.FP_modules) and not self.use_origin_pc_in_fp:
                l_features[i - 1] = self.FP_modules[i](l_xyz[i - 1], l_xyz[i], None, l_features[i])
            else:
                l_features[i - 1] = self.FP_modules[i](l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i])
        res = {'center_idx_origin': l_center_idx_origin,
               'sample_idx_origin': l_sample_idx_origin,
               'sa_features': [sa_features[1], sa_features[2], sa_features[3]],
               'fp_features': [l_features[2].unsqueeze(-1), l_features[1].unsqueeze(-1), l_features[0].unsqueeze(-1)]}
        return res


class _PointNet2SAModuleBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None
        #self.sas = None

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None) -> Tuple[Optional[Any], Any, Tensor, Tensor]:
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, N, C) tensor of the descriptors of the features
        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []
        xyz_trans = xyz.transpose(1, 2).contiguous()  # B x 3 x N
        center_idx = pointops.furthestsampling(xyz, self.npoint)  # B x npoint
        # sampled points
        new_xyz = pointops.gathering(
            xyz_trans,
            center_idx
        ).transpose(1, 2).contiguous() if self.npoint is not None else None
        # features of sampled points
        center_features = pointops.gathering(
            features,
            center_idx
        )
        # grouping local features
        sample_idx_list = []  # list of (B , npoint, nsample)
        for i in range(len(self.groupers)):
            new_features, sample_idx = self.groupers[i](xyz, new_xyz, features, center_features)  # B x C x M x K
            new_features = self.mlps[i](new_features)  # B x C' x M x K
            new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # B x C' x M x 1
            new_features = new_features.squeeze(-1)  # B x C' x M
            # use attention
            #new_features = self.sas[i](new_features)
            new_features_list.append(new_features)
            sample_idx_list.append(sample_idx)
        sample_idx = torch.cat(sample_idx_list, dim=-1)
        return new_xyz, center_idx, sample_idx, torch.cat(new_features_list, dim=1)


class PointNet2SAModuleMSG(_PointNet2SAModuleBase):
    r"""Pointnet set abstraction layer with multiscale grouping
    Parameters
    ----------
    npoint : int
        Number of features
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query or knn
    mlps : list of int32
        Spec of the pointnet_old before the global max_pool for each scale
    gp: int
        Number of the divided query map in self-attention layer
    bn : bool
        Use batchnorm
    use_xyz: bool
        use xyz only
    """

    def __init__(self, *, npoint: int, radii: List[float], nsamples: List[int], knn_dilation: int,
                 mlps: List[List[int]], gp: int,
                 bn: bool = True, use_xyz: bool = True):
        super().__init__()
        assert len(radii) == len(nsamples) == len(mlps)
        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        #self.sas = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointops.QueryAndGroup_Edge(radius, nsample, knn_dilation=knn_dilation, use_xyz=use_xyz,
                                            ret_sample_idx=True)  # EdgeConv
                if npoint is not None else pointops.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3
            self.mlps.append(pt_util.SharedMLP(mlp_spec, bn=bn))
            #self.sas.append(lp.GroupSALayer(mlp_spec[-1], gp))


class PointNet2SAModule(PointNet2SAModuleMSG):
    r"""Pointnet set abstraction layer
    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query or knn
    gp: int
        Number of the divided query map in self-attention layer
    mlp : list
        Spec of the pointnet_old before the global max_pool
    bn : bool
        Use batchnorm
    use_xyz: bool
        use xyz only
    """

    def __init__(self, *, mlp: List[int], npoint: int = None, radius: float = None, nsample: int = None,
                 knn_dilation: int = 1, gp: int = None,
                 bn: bool = True, use_xyz: bool = True):
        super().__init__(mlps=[mlp], npoint=npoint, radii=[radius], nsamples=[nsample], knn_dilation=knn_dilation,
                         gp=gp, bn=bn, use_xyz=use_xyz)


class PointNet2FPModule(nn.Module):
    r"""Propagates the features of one set to another
    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm
    """

    def __init__(self, *, mlp: List[int], bn: bool = True):
        super().__init__()
        self.mlp = pt_util.SharedMLP(mlp, bn=bn)

    def forward(self, unknown: torch.Tensor, known: torch.Tensor, unknow_feats: torch.Tensor,
                known_feats: torch.Tensor) -> torch.Tensor:
        r"""
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propagated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propagated
        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """
        if known is not None:
            dist, idx = pointops.nearestneighbor(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_feats = pointops.interpolation(known_feats, idx, weight)
        else:
            interpolated_feats = known_feats.expand(*known_feats.size()[0:2], unknown.size(1))

        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)
        else:
            new_features = interpolated_feats
        new_features = self.mlp(new_features.unsqueeze(-1)).squeeze(-1)
        return new_features


if __name__ == "__main__":
    batch_size = 2
    # l_center_idx = []  # list of B x npoint
    # l_sample_idx = []  # list of B x npoint x nsample
    # li_center_idx = [[0, 2, 4, 6, 8, 10, 12, 14], [0, 2, 4, 6, 8, 10, 12, 14]]
    # li_sample_idx = [[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]],
    #                  [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]]
    # li_center_idx = torch.LongTensor(li_center_idx)
    # li_sample_idx = torch.LongTensor(li_sample_idx)
    # l_center_idx.append(li_center_idx)
    # l_sample_idx.append(li_sample_idx)
    # li_center_idx = [[0, 2, 4, 6], [0, 2, 4, 6]]
    # li_sample_idx = [[[0, 1], [2, 3], [4, 5], [6, 7]], [[0, 1], [2, 3], [4, 5], [6, 7]]]
    # li_center_idx = torch.LongTensor(li_center_idx)
    # li_sample_idx = torch.LongTensor(li_sample_idx)
    # l_center_idx.append(li_center_idx)
    # l_sample_idx.append(li_sample_idx)
    # li_center_idx = [[0, 2], [0, 2]]
    # li_sample_idx = [[[0, 1], [2, 3]], [[0, 1], [2, 3]]]
    # li_center_idx = torch.LongTensor(li_center_idx)
    # li_sample_idx = torch.LongTensor(li_sample_idx)
    # l_center_idx.append(li_center_idx)
    # l_sample_idx.append(li_sample_idx)
    # # get center idx and sample idx
    # l_center_idx_origin = [l_center_idx[0]]
    # l_sample_idx_origin = [l_sample_idx[0]]
    # for i in range(1, len(l_center_idx)):
    #     li_center_idx_origin = torch.gather(l_center_idx_origin[i-1], -1, l_center_idx[i])
    #     temp_l_center_idx_origin = l_center_idx_origin[i - 1].unsqueeze(1)
    #     temp_l_center_idx_origin = temp_l_center_idx_origin.repeat(1, l_sample_idx[i].shape[1], 1)
    #     li_sample_idx_origin = torch.gather(temp_l_center_idx_origin, -1, l_sample_idx[i])
    #     l_center_idx_origin.append(li_center_idx_origin)
    #     l_sample_idx_origin.append(li_sample_idx_origin)
    # print(l_center_idx_origin)
    # print(l_sample_idx_origin)
