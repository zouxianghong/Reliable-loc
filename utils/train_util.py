import os
import argparse
import logging
import numpy as np
import random
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from libs.KNN_CUDA import knn_cuda
from utils.util import check_makedirs


def get_cfg(description, cfg_file):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', type=str, default=cfg_file, help='config file')

    args = parser.parse_args()
    print('args.config: {}'.format(args.config))
    cfg = yaml.safe_load(open(args.config, 'r'))
    return cfg


def set_seed(seed=None):
    if seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        print("no seed setting!!!")


def get_logger(log_dir):
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.propagate = False
    check_makedirs(log_dir)
    log_filename = os.path.join(log_dir, "train.log")
    file_handler = logging.FileHandler(filename=log_filename)
    logger.addHandler(file_handler)
    return logger


def get_model(cfg):
    if cfg['model_type'] == "pptnet":
        from place_recognition.pptnet_origin.models import pptnet
        model = pptnet.Network(param=cfg, use_normalize=False)
    elif cfg['model_type'] == "pptnet_l2_norm":
        from place_recognition.pptnet_origin.models import pptnet
        model = pptnet.Network(param=cfg, use_normalize=True)
    elif cfg['model_type'] == "pointnet_vlad":
        from place_recognition.pointnet_vlad import PointNetVlad
        model = PointNetVlad.PointNetVlad(global_feat=True, feature_transform=True, max_pool=False,
                                          output_dim=256, num_points=4096)
    elif cfg['model_type'] == "patch_aug_net":
        from place_recognition.patch_aug_net.models import patch_aug_net
        model = patch_aug_net.Network(param=cfg, use_a2a_recon=True, use_l2_norm=True)
    elif cfg['model_type'] == 'minkloc3d_v2':
        from place_recognition.Minkloc3D_V2.models.model_factory import model_factory
        from place_recognition.Minkloc3D_V2.misc.utils import TrainingParams
        params = TrainingParams(cfg['config'], cfg['model_config'])
        model = model_factory(params.model_params)
    elif cfg['model_type'] == 'egonn':
        from place_recognition.EgoNN.models.model_factory import model_factory
        from place_recognition.EgoNN.misc.utils import TrainingParams
        params = TrainingParams(cfg['config'], cfg['model_config'])
        model = model_factory(params.model_params)
    elif cfg['model_type'] == 'lcdnet':
        from place_recognition.LCDNet.get_models_cfg import get_model
        from collections import OrderedDict
        saved_params = torch.load(cfg['weight'], map_location='cpu')
        exp_cfg = saved_params['config']
        exp_cfg.batch_size = 6
        exp_cfg.loop_file = 'loop_GT_4m'
        exp_cfg.head = 'UOTHead'
        model = get_model(exp_cfg, is_training=False)
        renamed_dict = OrderedDict()
        for key in saved_params['state_dict']:
            if not key.startswith('module'):
                renamed_dict = saved_params['state_dict']
                break
            else:
                renamed_dict[key[7:]] = saved_params['state_dict'][key]
        # Convert shape from old OpenPCDet
        if renamed_dict['backbone.backbone.conv_input.0.weight'].shape != model.state_dict()[
            'backbone.backbone.conv_input.0.weight'].shape:
            for key in renamed_dict:
                if key.startswith('backbone.backbone.conv') and key.endswith('weight'):
                    if len(renamed_dict[key].shape) == 5:
                        renamed_dict[key] = renamed_dict[key].permute(-1, 0, 1, 2, 3)
        res = model.load_state_dict(renamed_dict, strict=True)
        if len(res[0]) > 0:
            print(f"WARNING: MISSING {len(res[0])} KEYS, MAYBE WEIGHTS LOADING FAILED")
    elif cfg['model_type'] == 'logg3d_net':
        from place_recognition.LoGG3DNet.models import model_factory
        model = model_factory.model_factory('logg3d1k')
        print('Loading weights: {}'.format(cfg['weight']))
        checkpoint = torch.load(cfg['weight'])
        model.load_state_dict(checkpoint['model_state_dict'])
    elif cfg['model_type'] == 'correlation':
        from rerank.sdp_rerank.models import rerank_net
        model = rerank_net.RerankNetCorrelation(cfg)
    elif cfg['model_type'] == 'overlap':
        from rerank.sdp_rerank.models import rerank_net
        model = rerank_net.RerankNetOverlap(cfg)
    elif cfg['model_type'] == 'metrics':
        from rerank.sdp_rerank.models import rerank_net
        model = rerank_net.RerankNetMetrics(cfg)
    elif cfg['model_type'] == 'metrics_top_k':
        from rerank.sdp_rerank.models import rerank_net
        model = rerank_net.RerankNetMetricsTopK(cfg)
    elif cfg['model_type'] == 'metrics_seq':
        from rerank.sdp_rerank.models import rerank_net
        model = rerank_net.RerankNetMetricsSeq(cfg)
    elif cfg['model_type'] == 'pose_est_matcher':
        from pose_estimation import pose_est_net
        model = pose_est_net.PoseEstMatcher(cfg)
    elif cfg['model_type'] == 'pose_est_net':
        from pose_estimation import pose_est_net
        model = pose_est_net.PoseEstNet(cfg)
    else:
        raise ValueError(f"Not a valid model!")
    return model


def load_model(model, weight_file):
    checkpoint = torch.load(weight_file)
    if 'state_dict_encoder' in checkpoint:
        model.load_state_dict(checkpoint['state_dict_encoder'])  # old model use 'state_dict_encoder' !
    else:
        model.load_state_dict(checkpoint)
    return model


def print_model(model, logger):
    logger.info("=> creating model ...")
    logger.info(model)
    parameters = model.parameters()
    total = sum([param.nelement() for param in parameters])
    logger.info("Number of parameter: %.4fM" % (total / 1e6))


def get_device(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in cfg["TRAIN_GPU"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def get_current_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def get_loss_func(config):
    from losses import pointnetvlad_loss
    from losses.focal_loss import BinaryFocalLoss
    from losses.multi_similarity_loss import MultiSimilarityLoss
    from losses.pose_est_loss import OTLoss, PPSLoss
    from losses.contrastive_loss import ContrastiveLoss
    from losses.truncated_smoothap import TruncatedSmoothAP
    if config['loss_type'] == 'quadruplet':
        return pointnetvlad_loss.quadruplet_loss
    elif config['loss_type'] == 'hphn_quadruplet':
        return pointnetvlad_loss.hphn_quadruplet_loss
    elif config['loss_type'] == 'contrastive':
        return pointnetvlad_loss.contrastive_loss
    elif config['loss_type'] == 'chamfer':
        return pointnetvlad_loss.chamfer_loss
    elif config['loss_type'] == 'patch_chamfer':
        return pointnetvlad_loss.patch_chamfer_loss
    elif config['loss_type'] == 'emd':
        return pointnetvlad_loss.emd_loss
    elif config['loss_type'] == 'patch_emd':
        return pointnetvlad_loss.patch_emd_loss
    elif config['loss_type'] == 'point_pair':
        return pointnetvlad_loss.point_pair_loss
    elif config['loss_type'] == 'triplet_custom':
        return pointnetvlad_loss.triplet_loss_wrapper
    elif config['loss_type'] == 'binary_cross_entropy' or config['loss_type'] == 'BCE':
        return nn.BCELoss(reduction='mean')
    elif config['loss_type'] == 'binary_focal':
        return BinaryFocalLoss(alpha=0.25, with_logit=False)
    elif config['loss_type'] == 'triplet_pytorch':
        return nn.TripletMarginLoss(margin=0.5, p=2)
    elif config['loss_type'] == 'contrastive2':
        return ContrastiveLoss(margin=0.5)
    elif config['loss_type'] == 'L1':
        return nn.L1Loss()
    elif config['loss_type'] == 'multi_similarity':
        return MultiSimilarityLoss(thresh=0.5, margin=0.3, scale_pos=2.0, scale_neg=40.0)
    elif config['loss_type'] == 'optimal_transport':
        return OTLoss()
    elif config['loss_type'] == 'point_pairs':
        return PPSLoss(margin=0.75)
    elif config['loss_type'] == 'cross_entropy':
        return nn.CrossEntropyLoss(reduction='mean')
    elif config['loss_type'] == 'SmoothAP':
        return TruncatedSmoothAP(tau1=0.01, similarity='cosine', positives_per_query=5)
    elif config['loss_type'] == 'relative_order':
        from losses.relative_order_loss import RelativeOrderLoss
        return RelativeOrderLoss()
    else:
        raise ValueError(f"Not a valid loss function!")


def get_optimizer(model, cfg):
    if cfg["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg['base_learning_rate'], weight_decay=cfg['adam_decay'])
    elif cfg["optimizer"] == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg['base_learning_rate'], weight_decay=cfg['adamw_decay'])
    elif cfg["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg['base_learning_rate'], weight_decay=cfg['sgd_decay'], momentum=cfg['sgd_momentum'])
    else:
        raise ValueError(f"Not a valid optimizer!")


def get_lr_scheduler(cfg, optimizer):
    if cfg['lr_decay_type'] == 'step':
        return StepLR(optimizer, step_size=cfg['lr_step_size'], gamma=cfg['lr_step_gamma'])
    elif cfg['lr_decay_type'] == 'cosine':
        return CosineAnnealingLR(optimizer, cfg['max_epoch'])
    else:
        return None


def save_model(model, epoch, iter, optimizer, cfg, logger, i=None, copy_to_event_dir=False):
    if isinstance(model, nn.DataParallel):
        model_to_save = model.module
    else:
        model_to_save = model

    check_makedirs(cfg["save_path"])
    if i is not None:
        save_name = os.path.join(cfg["save_path"], "train_epoch_{}_iter{}.pth".format(str(epoch), str(i)))
    else:
        save_name = os.path.join(cfg["save_path"], "train_epoch_{}_end.pth".format(str(epoch)))
    obj_to_save = {'epoch': epoch, 'iter': iter, 'optimizer': optimizer.state_dict(),
                   'state_dict_encoder': model_to_save.state_dict(),  # old model use 'state_dict_encoder' !
                   'config': cfg
                   }
    torch.save(obj_to_save, save_name)
    logger.info("Model Saved As {}".format(save_name))
    if copy_to_event_dir:
        import shutil
        shutil.copyfile(save_name, os.path.join(cfg['event_dir'], "train_epoch_x_end.pth"))


def log_model_grad(model):
    for name, params in model.named_parameters():
        if params.data is None or params.grad is None:
            continue
        print('-->name:', name, '-->grad_requirs:', params.requires_grad, '--weight', torch.mean(params.data), '-->grad_value:', torch.mean(params.grad))


def get_flops(model, tensor):
    from fvcore.nn import FlopCountAnalysis
    flops = FlopCountAnalysis(model, tensor).total() / (10**9)  # GFLOPs
    return flops


def get_num_param(model):
    from fvcore.nn import parameter_count_table
    num_param = parameter_count_table(model)
    # num_param = sum([param.nelement() for param in model.parameters()]) / 1e6
    return num_param


def nn_dist(c):
    # c: m x 3, or b x m x 3; Return: m x m, or b x m x m
    if len(c.shape) == 2:
        c1 = torch.unsqueeze(c, dim=1)
        c2 = c[None, ...]
    elif len(c.shape) == 3:
        c1 = torch.unsqueeze(c, dim=2)
        c2 = c[:, None, ...]
    return torch.sum((c1 - c2)**2, dim=-1) ** 0.5


def nn_dist_np(c):
    """ c: n x d """
    c1 = c[:, None, :]  # n x 1 x d
    c2 = c[None, ...]  # 1 x n x d
    return np.sum((c1 - c2)**2, axis=-1) ** 0.5  # n x n


def nn_angle(c, k=3):
    # c: m x 3, or b x m x 3
    knn = knn_cuda.KNN(k=k+1, transpose_mode=True)
    if len(c.shape) == 2:
        c = c.unsqueeze(0)  # 1 x m x 3
    # nearest k neighborhood
    _, index = knn(c, c)  # b x m x (k+1)
    index = index[..., 1:]  # b x m x k

    # cos_angle = []
    # for i in range(index.shape[0]):
    #     c_i = c[i]  # m x 3
    #     c0 = c_i[:, None, :]  # m x 1 x 3
    #     c1 = c_i[None, ...]  # 1 x m x 3
    #     index_i = index[i]  # m x k
    #     c2 = c_i[index_i]  # m x k x 3
    #     c01 = c1 - c0  # m x m x 3
    #     c02 = c2 - c0  # m x k x 3
    #     c01 = c01.unsqueeze(0)  # 1 x m x m x 3
    #     c02 = c02.unsqueeze(0).transpose(0, 2).contiguous()  # k x m x 1 x 3
    #     angle_i = F.cosine_similarity(c01, c02, dim=-1).unsqueeze(0)  # 1 x k x m x m
    #     cos_angle.append(angle_i)
    # cos_angle = torch.cat(cos_angle, dim=0)  # b x k x m x m

    c0 = c[..., None, :]  # b x m x 1 x 3
    c1 = c[:, None, ...]  # b x 1 x m x 3
    c2 = []
    for i in range(index.shape[0]):
        c2_i = c[i][index[i]]  # m x k x 3
        c2.append(c2_i.unsqueeze(0))  # 1 x m x k x 3
    c2 = torch.cat(c2, dim=0)  # b x m x k x 3
    c01 = c1 - c0  # b x m x m x 3
    c02 = c2 - c0  # b x m x k x 3
    c01 = c01.unsqueeze(1)  # b x 1 x m x m x 3
    c02 = c02.unsqueeze(0).permute(1, 3, 2, 0, 4)  # b x k x m x 1 x 3
    cos_angle = F.cosine_similarity(c01, c02, dim=-1)  # b x k x m x m
    return cos_angle
