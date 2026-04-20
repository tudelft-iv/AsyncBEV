# ------------------------------------------------------------------------
# Copyright (c) 2023 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

from distutils.command.build import build
import enum
from turtle import down
import math
import copy
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mmcv
from mmcv.cnn import ConvModule, build_conv_layer
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
from mmcv.runner import BaseModule, force_fp32
from mmcv.cnn import xavier_init, constant_init, kaiming_init
from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean, build_bbox_coder)
from mmdet.models.utils import build_transformer
from mmdet.models import HEADS, build_loss
from mmdet.models.utils import NormedLinear
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.models.utils.clip_sigmoid import clip_sigmoid
from mmdet3d.models import builder
from mmdet3d.core import (circle_nms, draw_heatmap_gaussian, gaussian_radius,
                          xywhr2xyxyr)
from einops import rearrange
import collections

from functools import reduce
from projects.CMT.mmdet3d_plugin.core.bbox.util import normalize_bbox

### For AsyncBEV
from projects.CMT.mmdet3d_plugin.models.async_modules import FlowEmbedder, Voxelizer
from projects.CMT.mmdet3d_plugin.models.async_modules import FlowUNet
from projects.CMT.mmdet3d_plugin.models.async_modules import SimpleBEVEncoder
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion


def pos2embed(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = 2 * (dim_t // 2) / num_pos_feats + 1
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x), dim=-1)
    return posemb


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, groups, eps):
        ctx.groups = groups
        ctx.eps = eps
        N, C, L = x.size()
        x = x.view(N, groups, C // groups, L)
        mu = x.mean(2, keepdim=True)
        var = (x - mu).pow(2).mean(2, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1) * y.view(N, C, L) + bias.view(1, C, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        groups = ctx.groups
        eps = ctx.eps

        N, C, L = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1)
        g = g.view(N, groups, C//groups, L)
        mean_g = g.mean(dim=2, keepdim=True)
        mean_gy = (g * y).mean(dim=2, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx.view(N, C, L), (grad_output * y.view(N, C, L)).sum(dim=2).sum(dim=0), grad_output.sum(dim=2).sum(
            dim=0), None, None


class GroupLayerNorm1d(nn.Module):

    def __init__(self, channels, groups=1, eps=1e-6):
        super(GroupLayerNorm1d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.groups = groups
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.groups, self.eps)


@HEADS.register_module()
class SeparateTaskHead(BaseModule):
    """SeparateHead for CenterHead.

    Args:
        in_channels (int): Input channels for conv_layer.
        heads (dict): Conv information.
        head_conv (int): Output channels.
            Default: 64.
        final_kernal (int): Kernal size for the last conv layer.
            Deafult: 1.
        init_bias (float): Initial bias. Default: -2.19.
        conv_cfg (dict): Config of conv layer.
            Default: dict(type='Conv2d')
        norm_cfg (dict): Config of norm layer.
            Default: dict(type='BN2d').
        bias (str): Type of bias. Default: 'auto'.
    """

    def __init__(self,
                 in_channels,
                 heads,
                 groups=1,
                 head_conv=64,
                 final_kernel=1,
                 init_bias=-2.19,
                 init_cfg=None,
                 **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super(SeparateTaskHead, self).__init__(init_cfg=init_cfg)
        self.heads = heads
        self.groups = groups
        self.init_bias = init_bias
        for head in self.heads:
            classes, num_conv = self.heads[head]

            conv_layers = []
            c_in = in_channels
            for i in range(num_conv - 1):
                conv_layers.extend([
                    nn.Conv1d(
                        c_in * groups,
                        head_conv * groups,
                        kernel_size=final_kernel,
                        stride=1,
                        padding=final_kernel // 2,
                        groups=groups,
                        bias=False),
                    GroupLayerNorm1d(head_conv * groups, groups=groups),
                    nn.ReLU(inplace=True)
                ])
                c_in = head_conv

            conv_layers.append(
                nn.Conv1d(
                    head_conv * groups,
                    classes * groups,
                    kernel_size=final_kernel,
                    stride=1,
                    padding=final_kernel // 2,
                    groups=groups,
                    bias=True))
            conv_layers = nn.Sequential(*conv_layers)

            self.__setattr__(head, conv_layers)

            if init_cfg is None:
                self.init_cfg = dict(type='Kaiming', layer='Conv1d')

    def init_weights(self):
        """Initialize weights."""
        super().init_weights()
        for head in self.heads:
            if head == 'cls_logits':
                self.__getattr__(head)[-1].bias.data.fill_(self.init_bias)

    def forward(self, x):
        """Forward function for SepHead.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [N, B, query, C].

        Returns:
            dict[str: torch.Tensor]: contains the following keys:

                -reg （torch.Tensor): 2D regression value with the \
                    shape of [N, B, query, 2].
                -height (torch.Tensor): Height value with the \
                    shape of [N, B, query, 1].
                -dim (torch.Tensor): Size value with the shape \
                    of [N, B, query, 3].
                -rot (torch.Tensor): Rotation value with the \
                    shape of [N, B, query, 2].
                -vel (torch.Tensor): Velocity value with the \
                    shape of [N, B, query, 2].
        """
        N, B, query_num, c1 = x.shape
        x = rearrange(x, "n b q c -> b (n c) q")
        ret_dict = dict()
        
        for head in self.heads:
             head_output = self.__getattr__(head)(x)
             ret_dict[head] = rearrange(head_output, "b (n c) q -> n b q c", n=N)

        return ret_dict


@HEADS.register_module()
class CmtHead(BaseModule):

    def __init__(self,
                 in_channels,
                 num_query=900,
                 hidden_dim=128,
                 depth_num=64,
                 ## AsyncBEV parameters
                 bev_h = 180,
                 bev_w = 180,
                 async_warp_modality = None,
                 learned_bev_warping = None, # 'vel_with_loss',
                 flow_modal = None, # 'multi'
                 use_time_encoding=True,
                 use_emc_pe_encoding=True,
                 delta_t_noise = False,
                 vis_output=None,
                 ##
                 norm_bbox=True,
                 downsample_scale=8,
                 scalar=10,
                 noise_scale=1.0,
                 noise_trans=0.0,
                 dn_weight=1.0,
                 split=0.75,
                 train_cfg=None,
                 test_cfg=None,
                 common_heads=dict(
                     center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)
                 ),
                 tasks=[
                    dict(num_class=1, class_names=['car']),
                    dict(num_class=2, class_names=['truck', 'construction_vehicle']),
                    dict(num_class=2, class_names=['bus', 'trailer']),
                    dict(num_class=1, class_names=['barrier']),
                    dict(num_class=2, class_names=['motorcycle', 'bicycle']),
                    dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
                 ],
                 transformer=None,
                 bbox_coder=None,
                 loss_cls=dict(
                     type="FocalLoss",
                     use_sigmoid=True,
                     reduction="mean",
                     gamma=2, alpha=0.25, loss_weight=1.0
                 ),
                 loss_bbox=dict(
                    type="L1Loss",
                    reduction="mean",
                    loss_weight=0.25,
                 ),
                 loss_heatmap=dict(
                     type="GaussianFocalLoss",
                     reduction="mean"
                 ),
                 separate_head=dict(
                     type='SeparateMlpHead', init_bias=-2.19, final_kernel=3),
                 init_cfg=None,
                 **kwargs):
        assert init_cfg is None
        super(CmtHead, self).__init__(init_cfg=init_cfg)
        self.num_classes = [len(t["class_names"]) for t in tasks]
        self.class_names = [t["class_names"] for t in tasks]
        self.hidden_dim = hidden_dim
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_query = num_query
        self.in_channels = in_channels
        self.depth_num = depth_num
        self.norm_bbox = norm_bbox
        self.downsample_scale = downsample_scale
        self.scalar = scalar
        self.bbox_noise_scale = noise_scale
        self.bbox_noise_trans = noise_trans
        self.dn_weight = dn_weight
        self.split = split

        ## AsyncBEV parameters
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_warping = async_warp_modality
        self.flow_modal = flow_modal
        self.pc_range = bbox_coder.pc_range
        self.real_w = self.pc_range[3]-self.pc_range[0]
        self.real_h = self.pc_range[4]-self.pc_range[1]
        self.use_time_encoding = use_time_encoding
        self.use_emc_pe_encoding = use_emc_pe_encoding
        self.vis_output = vis_output
        self.delta_t_noise = delta_t_noise
        self.vis_data = {}
        if self.bev_warping:
            assert learned_bev_warping in ['motion', 'vel', 'gt', 'motion_with_loss', 'vel_with_loss', None]
            self.learned_temporal_bev_warping = learned_bev_warping
            ### temp_encoding_dec is not implementated yet
            if self.learned_temporal_bev_warping == 'vel_with_loss' or self.learned_temporal_bev_warping =='vel':
                if self.use_time_encoding:
                    self.timestamp_encoder = nn.Sequential(
                        nn.Linear(1, 4),
                        nn.ReLU(inplace=True),
                        nn.Linear(4, 4)
                    )
                if self.use_emc_pe_encoding:
                    self.bev_pose_encoder = nn.Sequential(
                        nn.Linear(2, 4),
                        nn.ReLU(inplace=True),
                        nn.Linear(4, 4)
                    )
                if self.flow_modal == 'multi':
                    self.simple_bev_encoder = SimpleBEVEncoder(
                        H_bev=self.bev_h,
                        W_bev=self.bev_w,
                        Z_bev=8,
                        pc_range=self.pc_range,
                        embedding_dim=self.hidden_dim,
                        output_dim=self.hidden_dim,
                    )
                    self.bev_embed_proj = nn.Sequential(
                        nn.Linear(self.hidden_dim*2, self.hidden_dim // 2),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.hidden_dim // 2, self.hidden_dim // 2),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.hidden_dim // 2, 8),
                    )
                elif self.flow_modal == 'same':
                    if self.bev_warping == 'cam':
                        self.simple_bev_encoder = SimpleBEVEncoder(
                            H_bev=self.bev_h,
                            W_bev=self.bev_w,
                            Z_bev=8,
                            pc_range=self.pc_range,
                            embedding_dim=self.hidden_dim,
                            output_dim=self.hidden_dim,
                        )
                    self.bev_embed_proj = nn.Sequential(
                        nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.hidden_dim // 2, self.hidden_dim // 2),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.hidden_dim // 2, 8),
                    )

                self.learned_vel_estimator = FlowUNet(
                    input_dim=8 + self.use_emc_pe_encoding * 4 + self.use_time_encoding * 4,
                    output_dim =2,
                    bev_size = (self.bev_h, self.bev_w),
                )
                self.pe_voxelizer =  Voxelizer(
                    voxel_size = [ self.real_h/self.bev_h, self.real_w/self.bev_w, self.pc_range[5] - self.pc_range[2]],
                    point_cloud_range = self.pc_range,
                )
            if self.learned_temporal_bev_warping == 'motion_with_loss' or self.learned_temporal_bev_warping == 'motion':
                if self.use_time_encoding:
                    self.timestamp_encoder = nn.Sequential(
                        nn.Linear(1, 4),
                        nn.ReLU(inplace=True),
                        nn.Linear(4, 4)
                    )
                if self.use_emc_pe_encoding:
                    self.bev_pose_encoder = nn.Sequential(
                        nn.Linear(2, 4),
                        nn.ReLU(inplace=True),
                        nn.Linear(4, 4)
                    )
                if self.flow_modal == 'multi':
                    self.simple_bev_encoder = SimpleBEVEncoder(
                        H_bev=self.bev_h,
                        W_bev=self.bev_w,
                        Z_bev=8,
                        pc_range=self.pc_range,
                        embedding_dim=self.hidden_dim,
                        output_dim=self.hidden_dim,
                    )
                    self.bev_embed_proj = nn.Sequential(
                        nn.Linear(self.hidden_dim*2, self.hidden_dim // 2),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.hidden_dim // 2, self.hidden_dim // 2),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.hidden_dim // 2, 8),
                    )
                elif self.flow_modal == 'same':
                    if self.bev_warping == 'cam':
                        self.simple_bev_encoder = SimpleBEVEncoder(
                            H_bev=self.bev_h,
                            W_bev=self.bev_w,
                            Z_bev=8,
                            pc_range=self.pc_range,
                            embedding_dim=self.hidden_dim,
                            output_dim=self.hidden_dim,
                        )
                    self.bev_embed_proj = nn.Sequential(
                        nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.hidden_dim // 2, self.hidden_dim // 2),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.hidden_dim // 2, 8),
                    )

                self.learned_motion_estimator = FlowUNet(
                    input_dim=8 + self.use_emc_pe_encoding * 4 + self.use_time_encoding * 4,
                    output_dim =2,
                    bev_size = (self.bev_h, self.bev_w),
                )
                self.pe_voxelizer =  Voxelizer(
                    voxel_size = [ self.real_h/self.bev_h, self.real_w/self.bev_w, self.pc_range[5] - self.pc_range[2]],
                    point_cloud_range = self.pc_range,
                )
            if self.learned_temporal_bev_warping == 'gt' or self.learned_temporal_bev_warping == 'vel_with_loss' or self.learned_temporal_bev_warping == 'motion_with_loss':
                self.flow_embedder = FlowEmbedder(
                    voxel_size=[self.real_h/self.bev_h, self.real_w/self.bev_w, self.pc_range[5] - self.pc_range[2]],
                    point_cloud_range=self.pc_range,
                    pseudo_image_dims=[self.bev_h, self.bev_w])

        loss_voxel_flow_dict = kwargs.get('loss_voxel_flow', None)
        if loss_voxel_flow_dict is not None:
            self.loss_voxel_flow = build_loss(loss_voxel_flow_dict)
        else:
            self.loss_voxel_flow = None
        ##
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_heatmap = build_loss(loss_heatmap)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.fp16_enabled = False
           
        self.shared_conv = ConvModule(
            in_channels,
            hidden_dim,
            kernel_size=3,
            padding=1,
            conv_cfg=dict(type="Conv2d"),
            norm_cfg=dict(type="BN2d")
        )
        
        # transformer
        self.transformer = build_transformer(transformer)
        self.reference_points = nn.Embedding(num_query, 3)
        self.bev_embedding = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.rv_embedding = nn.Sequential(
            nn.Linear(self.depth_num * 3, self.hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim)
        )
        # task head
        self.task_heads = nn.ModuleList()
        for num_cls in self.num_classes:
            heads = copy.deepcopy(common_heads)
            heads.update(dict(cls_logits=(num_cls, 2)))
            separate_head.update(
                in_channels=hidden_dim,
                heads=heads, num_cls=num_cls,
                groups=transformer.decoder.num_layers
            )
            self.task_heads.append(builder.build_head(separate_head))

        # assigner
        if train_cfg:
            self.assigner = build_assigner(train_cfg["assigner"])
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

    def init_weights(self):
        super(CmtHead, self).init_weights()
        nn.init.uniform_(self.reference_points.weight.data, 0, 1)

    @property
    def coords_bev(self):
        cfg = self.train_cfg if self.train_cfg else self.test_cfg
        x_size, y_size = (
            cfg['grid_size'][1] // self.downsample_scale,
            cfg['grid_size'][0] // self.downsample_scale
        )
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_y, batch_x = torch.meshgrid(*[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = (batch_x + 0.5) / x_size
        batch_y = (batch_y + 0.5) / y_size
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)
        coord_base = coord_base.view(2, -1).transpose(1, 0) # (H*W, 2)
        return coord_base

    def prepare_for_dn(self, batch_size, reference_points, img_metas):
        if self.training:
            targets = [torch.cat((img_meta['gt_bboxes_3d']._data.gravity_center, img_meta['gt_bboxes_3d']._data.tensor[:, 3:]),dim=1) for img_meta in img_metas ]
            labels = [img_meta['gt_labels_3d']._data for img_meta in img_metas ]
            known = [(torch.ones_like(t)).cuda() for t in labels]
            know_idx = known
            unmask_bbox = unmask_label = torch.cat(known)
            known_num = [t.size(0) for t in targets]
            labels = torch.cat([t for t in labels])
            boxes = torch.cat([t for t in targets])
            batch_idx = torch.cat([torch.full((t.size(0), ), i) for i, t in enumerate(targets)])

            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)
            # add noise
            groups = min(self.scalar, self.num_query // max(known_num))
            known_indice = known_indice.repeat(groups, 1).view(-1)
            known_labels = labels.repeat(groups, 1).view(-1).long().to(reference_points.device)
            known_labels_raw = labels.repeat(groups, 1).view(-1).long().to(reference_points.device)
            known_bid = batch_idx.repeat(groups, 1).view(-1)
            known_bboxs = boxes.repeat(groups, 1).to(reference_points.device)
            known_bbox_center = known_bboxs[:, :3].clone()
            known_bbox_scale = known_bboxs[:, 3:6].clone()
            
            if self.bbox_noise_scale > 0:
                diff = known_bbox_scale / 2 + self.bbox_noise_trans
                rand_prob = torch.rand_like(known_bbox_center) * 2 - 1.0
                known_bbox_center += torch.mul(rand_prob,
                                            diff) * self.bbox_noise_scale
                known_bbox_center[..., 0:1] = (known_bbox_center[..., 0:1] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
                known_bbox_center[..., 1:2] = (known_bbox_center[..., 1:2] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
                known_bbox_center[..., 2:3] = (known_bbox_center[..., 2:3] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
                known_bbox_center = known_bbox_center.clamp(min=0.0, max=1.0)
                mask = torch.norm(rand_prob, 2, 1) > self.split
                known_labels[mask] = sum(self.num_classes)

            single_pad = int(max(known_num))
            pad_size = int(single_pad * groups)
            padding_bbox = torch.zeros(pad_size, 3).to(reference_points.device)
            padded_reference_points = torch.cat([padding_bbox, reference_points], dim=0).unsqueeze(0).repeat(batch_size, 1, 1)

            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(groups)]).long()
            if len(known_bid):
                padded_reference_points[(known_bid.long(), map_known_indice)] = known_bbox_center.to(reference_points.device)

            tgt_size = pad_size + self.num_query
            attn_mask = torch.ones(tgt_size, tgt_size).to(reference_points.device) < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(groups):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                if i == groups - 1:
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True

            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'known_labels_raw': known_labels_raw,
                'know_idx': know_idx,
                'pad_size': pad_size
            }
            
        else:
            padded_reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)
            attn_mask = None
            mask_dict = None

        return padded_reference_points, attn_mask, mask_dict
    def temporal_bev_warping(self, img_feats, pts_feats, coords_3d_reshape, img_metas):
        """
        coords_3d: [B, H, W, D, 3]
        """
        device = coords_3d_reshape.device
        bs, num_query, C = coords_3d_reshape.shape

        ## generate the matrix from async to sync
        if self.bev_warping == 'cam':
            ## the ego pose in global coordinate is the ego pose of 'CAM_BACK_LEFT'
            T_e2g_s_list = []
            T_l2e_s_list = []
            T_e2g_list = []
            delta_t_list = []

            for img_meta in img_metas:
                async_metas = img_meta['async_metas']
                timestamp = img_meta['timestamp']
                async_timestamp = async_metas[4]['timestamp']
                delta_t = timestamp - async_timestamp
                delta_t_list.append(delta_t)

                ego2global_translation = async_metas[4]['ego2global_translation']
                ego2global_rotation = async_metas[4]['ego2global_rotation']
                T_e2g_s = torch.tensor(transform_matrix(ego2global_translation, Quaternion(ego2global_rotation)))
                T_e2g_s_list.append(T_e2g_s)

                lidar2ego_translation = img_meta['lidar2ego_translation']
                lidar2ego_rotation = img_meta['lidar2ego_rotation']
                T_l2e_s = torch.tensor(transform_matrix(lidar2ego_translation, Quaternion(lidar2ego_rotation)))
                T_l2e_s_list.append(T_l2e_s)

                key_ego2global_translation = img_meta['ego2global_translation']
                key_ego2global_rotation = img_meta['ego2global_rotation']
                T_e2g = torch.tensor(transform_matrix(key_ego2global_translation, Quaternion(key_ego2global_rotation)))
                # print('T_e2g:', T_e2g.shape)
                # print('T_e2g_s:', T_e2g_s.shape)
                # print('T_l2e_s:', T_l2e_s.shape)

                T_e2g_list.append(T_e2g)

            # print('len T_e2g_s_list:', len(T_e2g_s_list))
            T_e2g_s_batched = torch.stack(T_e2g_s_list, dim=0).to(device).view(bs, 4, 4)
            T_l2e_s_batched = torch.stack(T_l2e_s_list, dim=0).to(device).view(bs, 4, 4)
            T_e2g_batched = torch.stack(T_e2g_list, dim=0).to(device).view(bs, 4, 4)
            delta_t_batched = torch.tensor(delta_t_list).to(device).view(bs, 1, 1)

        elif self.bev_warping == 'lidar':
            T_e2g_s_list = []
            T_l2e_s_list = []
            T_e2g_list = []
            delta_t_list = []
            for img_meta in img_metas:
                async_metas = img_meta['async_metas']

                timestamp = img_meta['timestamp']
                async_timestamp = async_metas['timestamp']
                delta_t = timestamp - async_timestamp
                delta_t_list.append(delta_t)
                # print('in bev encoders')
                # print('delta_t:', delta_t)

                ego2global_translation = async_metas['ego2global_translation']
                ego2global_rotation = async_metas['ego2global_rotation']

                T_e2g_s = torch.tensor(transform_matrix(ego2global_translation, Quaternion(ego2global_rotation)))
                T_e2g_s_list.append(T_e2g_s)

                lidar2ego_translation = img_meta['lidar2ego_translation']
                lidar2ego_rotation = img_meta['lidar2ego_rotation']
                T_l2e_s = torch.tensor(transform_matrix(lidar2ego_translation, Quaternion(lidar2ego_rotation)))
                T_l2e_s_list.append(T_l2e_s)

                key_ego2global_translation = img_meta['ego2global_translation']
                key_ego2global_rotation = img_meta['ego2global_rotation']
                T_e2g = torch.tensor(transform_matrix(key_ego2global_translation, Quaternion(key_ego2global_rotation)))
                T_e2g_list.append(T_e2g)

            T_e2g_s_batched = torch.stack(T_e2g_s_list, dim=0).to(device).view(bs,4, 4)
            T_l2e_s_batched = torch.stack(T_l2e_s_list, dim=0).to(device).view(bs,  4, 4)
            T_e2g_batched = torch.stack(T_e2g_list, dim=0).to(device).view(bs, 4, 4)
            delta_t_batched = torch.tensor(delta_t_list).to(device).view(bs, 1, 1)

        else:
            raise ValueError(f'Unrecognized bev warping modality: {self.bev_warping}')
        T_l2e_batched = T_l2e_s_batched.clone()
        matrix_async2sync =  torch.inverse(T_l2e_batched) @ torch.inverse(T_e2g_batched) @ T_e2g_s_batched @ T_l2e_s_batched

        if self.delta_t_noise:
            sigma = 0.1/3
            noise = torch.normal(0, sigma, delta_t_batched.shape).to(device)
            delta_t_batched = delta_t_batched + noise
        H = self.bev_h
        W = self.bev_w
        Z = self.pc_range[5] - self.pc_range[2]

        xs = torch.linspace(0.5, W - 0.5, W, dtype=T_e2g_batched.dtype, device=device).view(1, 1, W).expand(1, H, W) / W
        ys = torch.linspace(0.5, H - 0.5, H, dtype=T_e2g_batched.dtype, device=device).view(1, H, 1).expand(1, H, W) / H
        zs = torch.linspace(0.5, Z - 0.5, 1, dtype=T_e2g_batched.dtype, device=device).view(-1, 1, 1).expand(1, H, W) / Z

        ref_grid_3d = torch.stack((xs, ys, zs), dim=-1).to(device)
        ref_grid_3d[..., 0:1] = ref_grid_3d[..., 0:1] * self.real_w + self.pc_range[0]
        ref_grid_3d[..., 1:2] = ref_grid_3d[..., 1:2] * self.real_h + self.pc_range[1]
        ref_grid_3d[..., 2:3] = ref_grid_3d[..., 2:3] * Z + self.pc_range[2]
        ref_grid_3d = torch.cat((ref_grid_3d, torch.ones_like(ref_grid_3d[..., :1])), dim=-1).unsqueeze(-1).repeat(bs, 1, 1, 1, 1)

        ref_grid_3d_s = matrix_async2sync[:, None, None, ...] @ ref_grid_3d
        ref_grid_2d_s_ori = ref_grid_3d_s[..., :2, 0].clone()

        pts_bev_feats = pts_feats.view(bs, self.hidden_dim, self.bev_h * self.bev_w).permute(0, 2, 1).contiguous()
        if self.learned_temporal_bev_warping == 'vel_with_loss' or self.learned_temporal_bev_warping == 'vel' or self.learned_temporal_bev_warping == 'motion_with_loss' or self.learned_temporal_bev_warping == 'motion':
            if self.bev_warping == 'lidar':
                bev_embed = pts_bev_feats
                if self.flow_modal == 'multi' or self.flow_modal == 'cross':
                    img_simple_bev_embed = self.simple_bev_encoder(img_feats, img_metas)
                    cross_modal_bev_embed = img_simple_bev_embed
                    multi_modal_bev_embed = torch.cat([img_simple_bev_embed, pts_bev_feats], dim=-1)
            elif self.bev_warping == 'cam':
                img_simple_bev_embed = self.simple_bev_encoder(img_feats, img_metas)
                bev_embed = img_simple_bev_embed
                cross_modal_bev_embed = pts_bev_feats
                multi_modal_bev_embed = torch.cat([img_simple_bev_embed, pts_bev_feats], dim=-1)
            else:
                raise ValueError(f'Unrecognized bev warping modality: {self.bev_warping}')
            # print('img_simple_bev_embed:', img_simple_bev_embed.shape)

            condition_input = []

            if self.flow_modal == 'cross':
                bev_embed_encoding = self.bev_embed_proj(cross_modal_bev_embed)
            elif self.flow_modal == 'multi':
                bev_embed_encoding = self.bev_embed_proj(multi_modal_bev_embed)
            elif self.flow_modal == 'same':
                bev_embed_encoding = self.bev_embed_proj(bev_embed)
            else:
                raise ValueError(f'Unrecognized flow modality: {self.flow_modal}')
            condition_input.append(bev_embed_encoding)

            if self.use_emc_pe_encoding:
                ref_grid_encoding = self.bev_pose_encoder(ref_grid_2d_s_ori.view(bs, -1, 2).to(bev_embed.dtype))
                condition_input.append(ref_grid_encoding)

            if self.use_time_encoding:
                timestamp_encoding = self.timestamp_encoder(delta_t_batched).repeat(1, self.bev_h * self.bev_w, 1)
                condition_input.append(timestamp_encoding)

            condition_encoding = torch.cat(condition_input, dim=-1)
            # print('condition_encoding:', condition_encoding.shape)
            if self.learned_temporal_bev_warping == 'vel_with_loss' or self.learned_temporal_bev_warping == 'vel':
                ref_grid_2d_s_vel = self.learned_vel_estimator(condition_encoding)
                ref_grid_2d_s_flow_learned_batched = ref_grid_2d_s_vel * delta_t_batched
            elif self.learned_temporal_bev_warping == 'motion_with_loss' or self.learned_temporal_bev_warping == 'motion':
                ref_grid_2d_s_flow_learned_batched = self.learned_motion_estimator(condition_encoding)

            # print('coords_3d ll:', coords_3d.shape)
            # coords_3d_reshape = coords_3d.reshape(bs, B//bs, coords_H, coords_W, D, 4).reshape(bs, -1, 4)
            # print(coords_3d_reshape.shape)
            pe_voxel_info_list = self.pe_voxelizer(coords_3d_reshape)
            assert(len(pe_voxel_info_list) == bs)
            gt_flow_list = []
            pts_in_boxes_list = []
            flow_coords_pe_list = []
            for img_meta in img_metas:
                gt_flow_list.append(torch.from_numpy(img_meta['gt_flow_async2sync']).to(device))
                # print('shape:', img_meta['gt_flow_async2sync'].shape)
                pts_in_boxes_list.append(torch.from_numpy(img_meta['pts_in_boxes_async']).to(device))
                # print('shape:', img_meta['pts_in_boxes_async'].shape)
                # print(self.pc_range)
            if self.learned_temporal_bev_warping == 'vel_with_loss' or self.learned_temporal_bev_warping == 'motion_with_loss':
                flow_voxel_info_list, flow_pseudoimage_batch = self.flow_embedder(pts_in_boxes_list, gt_flow_list)
            else:
                flow_voxel_info_list, flow_pseudoimage_batch = None, torch.zeros(bs, 3, H, W).to(device)
            for pe_voxel_info, bev_flow in zip(pe_voxel_info_list, ref_grid_2d_s_flow_learned_batched):
                pe_voxel_coords = pe_voxel_info['voxel_coords']  # (N, 3), zyx

                pe_voxel_coords_flat = pe_voxel_coords[:,0] + pe_voxel_coords[:,1]*W + pe_voxel_coords[:,2]
                pe_flow = bev_flow[pe_voxel_coords_flat.to(torch.long)]

                pe_valid_pts_indexes = pe_voxel_info['point_idxes']
                flow_coords_pe = torch.zeros_like(coords_3d_reshape[0, :, :2]).to(device).to(bev_flow.dtype)
                flow_coords_pe[pe_valid_pts_indexes] = pe_flow
                flow_coords_pe_list.append(flow_coords_pe)
            predicted_flow_batched = torch.stack(flow_coords_pe_list, dim=0)
        else:
            ref_grid_2d_s_flow_learned_batched = None,
            flow_pseudoimage_batch = torch.zeros(bs, 3, H, W).to(device)
            flow_voxel_info_list = None
            predicted_flow_batched = torch.zeros(bs, num_query, 2).to(device).to(coords_3d_reshape.dtype)

        coords_3d_reshape_homo = torch.ones(bs, num_query, 4, 1).to(device)
        coords_3d_reshape_homo[...,:3,0] = coords_3d_reshape[...,:3]
        coords_3d_reshape_homo_async2sync = matrix_async2sync[:, None, ...].to(coords_3d_reshape_homo.dtype) @ coords_3d_reshape_homo

        coords_3d_reshape[...,:3] =  coords_3d_reshape_homo_async2sync[...,:3,0]
        coords_3d_reshape[...,0:2] += predicted_flow_batched

        flow_out = dict(
            delta_t = delta_t_batched,
            pred_flow = ref_grid_2d_s_flow_learned_batched,
            gt_flow = flow_pseudoimage_batch.permute(0,2,3,1).view(bs, self.bev_h*self.bev_w, -1)[..., :2],
            flow_voxel_info_list = flow_voxel_info_list,
        )
        if self.vis_output:
            self.vis_data.update(flow_out)
        return coords_3d_reshape, flow_out

    def _rv_pe(self, img_feats=None, pts_feats=None, img_metas=None):
        BN, C, H, W = img_feats.shape
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        ###
        # print('in rv_pe:')
        # print('BN, C, H, W:', BN, C, H, W) # [12, 256, 40, 100]
        # print('pad_h, pad_w:', pad_h, pad_w) # [640, 1600]
        # print('depth num:', self.depth_num)  # 64
        ###
        coords_h = torch.arange(H, device=img_feats[0].device).float() * pad_h / H
        coords_w = torch.arange(W, device=img_feats[0].device).float() * pad_w / W
        coords_d = 1 + torch.arange(self.depth_num, device=img_feats[0].device).float() * (self.pc_range[3] - 1) / self.depth_num
        coords_h, coords_w, coords_d = torch.meshgrid([coords_h, coords_w, coords_d])

        coords = torch.stack([coords_w, coords_h, coords_d, coords_h.new_ones(coords_h.shape)], dim=-1)
        coords[..., :2] = coords[..., :2] * coords[..., 2:3]
        
        imgs2lidars = np.concatenate([np.linalg.inv(meta['lidar2img']) for meta in img_metas])
        imgs2lidars = torch.from_numpy(imgs2lidars).float().to(coords.device)
        coords_3d = torch.einsum('hwdo, bco -> bhwdc', coords, imgs2lidars)
        # print('img2lidars:', imgs2lidars.shape) # [12, 4, 4]
        # print('coords_3d:', coords_3d.shape) # [12, 40, 100, 64, 3]
        if self.bev_warping == 'cam':
            BN, coords_H, coords_W, D, C = coords_3d.shape
            bs = pts_feats.shape[0]
            N = BN // bs
            coords_3d_reshape = coords_3d.reshape(bs, N, coords_H, coords_W, D, 4).reshape(bs, -1, 4)
            coords_3d_warped, flow_outs= self.temporal_bev_warping(img_feats, pts_feats, coords_3d_reshape, img_metas)
            coords_3d_warped = coords_3d_warped.view(bs, N, coords_H, coords_W, D, C).view(BN, coords_H, coords_W, D, C)
        else:
            coords_3d_warped = coords_3d
            flow_outs = None
        coords_3d = (coords_3d_warped[..., :3] - coords_3d_warped.new_tensor(self.pc_range[:3])[None, None, None, :] )\
                        / (coords_3d_warped.new_tensor(self.pc_range[3:]) - coords_3d_warped.new_tensor(self.pc_range[:3]))[None, None, None, :]
        return self.rv_embedding(coords_3d.reshape(*coords_3d.shape[:-2], -1)), flow_outs

    def _bev_pe(self, img_feats, pts_feats, img_metas):
        # bev_pos_embeds = self.bev_embedding(pos2embed(self.coords_bev.to(x.device), num_pos_feats=self.hidden_dim))
        bev_pe = self.coords_bev.to(pts_feats.device) #(H*W, 2) (0-1)
        bs = pts_feats.shape[0]
        # print('in bev pe before warping:', bev_pe.shape, bev_pe.dtype)
        bev_pe_batch = bev_pe.unsqueeze(0).repeat(bs, 1, 1) #(bs, H*W, 2)
        # print('in bev pe:', bev_pe_batch.shape)
        if self.bev_warping == 'lidar':
            bev_pe_batch = bev_pe_batch * (bev_pe_batch.new_tensor(self.pc_range[3:5]) - bev_pe_batch.new_tensor(self.pc_range[0:2]))[None, None, :]+ bev_pe_batch.new_tensor(self.pc_range[0:2])[None, None, :]
            bev_pe_batch = torch.cat([bev_pe_batch, torch.ones(bs, bev_pe.shape[0], 1).to(bev_pe.device)], dim=-1)  # (bs, H*W, 3)
            # print('bev_pe_batch1:', bev_pe_batch.shape, bev_pe_batch.dtype)
            bev_pe_batch, flow_outs = self.temporal_bev_warping(img_feats, pts_feats, bev_pe_batch, img_metas)
            bev_pe_batch = (bev_pe_batch[..., :2] - bev_pe_batch.new_tensor(self.pc_range[0:2])[None, None, :]) / (bev_pe_batch.new_tensor(self.pc_range[3:5]) - bev_pe_batch.new_tensor(self.pc_range[0:2]))[None, None, :]
            # (0, 1)
        else:
            flow_outs = None

        # print('bev_pe_batch2:', bev_pe_batch.shape, bev_pe_batch.dtype)
        return self.bev_embedding(pos2embed(bev_pe_batch, num_pos_feats=self.hidden_dim)), flow_outs

    def _bev_query_embed(self, ref_points, img_metas):
        bev_embeds = self.bev_embedding(pos2embed(ref_points, num_pos_feats=self.hidden_dim))
        return bev_embeds

    def _rv_query_embed(self, ref_points, img_metas):
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        lidars2imgs = np.stack([meta['lidar2img'] for meta in img_metas])
        lidars2imgs = torch.from_numpy(lidars2imgs).float().to(ref_points.device)
        imgs2lidars = np.stack([np.linalg.inv(meta['lidar2img']) for meta in img_metas])
        imgs2lidars = torch.from_numpy(imgs2lidars).float().to(ref_points.device)

        ref_points = ref_points * (ref_points.new_tensor(self.pc_range[3:]) - ref_points.new_tensor(self.pc_range[:3])) + ref_points.new_tensor(self.pc_range[:3])
        proj_points = torch.einsum('bnd, bvcd -> bvnc', torch.cat([ref_points, ref_points.new_ones(*ref_points.shape[:-1], 1)], dim=-1), lidars2imgs)
        
        proj_points_clone = proj_points.clone()
        z_mask = proj_points_clone[..., 2:3].detach() > 0
        proj_points_clone[..., :3] = proj_points[..., :3] / (proj_points[..., 2:3].detach() + z_mask * 1e-6 - (~z_mask) * 1e-6) 
        # proj_points_clone[..., 2] = proj_points.new_ones(proj_points[..., 2].shape) 
        
        mask = (proj_points_clone[..., 0] < pad_w) & (proj_points_clone[..., 0] >= 0) & (proj_points_clone[..., 1] < pad_h) & (proj_points_clone[..., 1] >= 0)
        mask &= z_mask.squeeze(-1)

        coords_d = 1 + torch.arange(self.depth_num, device=ref_points.device).float() * (self.pc_range[3] - 1) / self.depth_num
        proj_points_clone = torch.einsum('bvnc, d -> bvndc', proj_points_clone, coords_d)
        proj_points_clone = torch.cat([proj_points_clone[..., :3], proj_points_clone.new_ones(*proj_points_clone.shape[:-1], 1)], dim=-1)
        projback_points = torch.einsum('bvndo, bvco -> bvndc', proj_points_clone, imgs2lidars)

        projback_points = (projback_points[..., :3] - projback_points.new_tensor(self.pc_range[:3])[None, None, None, :] )\
                        / (projback_points.new_tensor(self.pc_range[3:]) - projback_points.new_tensor(self.pc_range[:3]))[None, None, None, :]
        
        rv_embeds = self.rv_embedding(projback_points.reshape(*projback_points.shape[:-2], -1))
        rv_embeds = (rv_embeds * mask.unsqueeze(-1)).sum(dim=1)
        return rv_embeds

    def query_embed(self, ref_points, img_metas):
        ref_points = inverse_sigmoid(ref_points.clone()).sigmoid()
        bev_embeds = self._bev_query_embed(ref_points, img_metas)
        rv_embeds = self._rv_query_embed(ref_points, img_metas)
        return bev_embeds, rv_embeds

    def forward_single(self, x, x_img, img_metas):
        """
            x: [bs c h w]
            return List(dict(head_name: [num_dec x bs x num_query * head_dim]) ) x task_num
        """
        ret_dicts = []
        x = self.shared_conv(x)

        ### Debug
        # print('Debug in CMT forward single:')
        # print('points_features:', x.shape) # [bs, 256, 180, 180]
        # print('x_img:', type(x_img), x_img.shape) # [12, 256, 40, 100]
        # print('type of img_metas:', type(img_metas), len(img_metas)) # list, 2
        # print('img_mets:', img_metas[0].keys()) # ['filename', 'ori_shape', 'img_shape', 'lidar2img', 'pad_shape', 'scale_factor', 'pcd_horizontal_flip', 'pcd_vertical_flip', '
        #                                         # box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx', 'pcd_scale_factor', 'pcd_rotation', 'pts_filename', 'transformation_3d_flo
        #                                         # w', 'gt_bboxes_3d', 'gt_labels_3d', 'input_shape']

        reference_points = self.reference_points.weight
        ###
        # print('reference_points:', reference_points.shape) # [900, 3]
        ###
        ###
        reference_points, attn_mask, mask_dict = self.prepare_for_dn(x.shape[0], reference_points, img_metas)

        ## prepare for dn
        # print('refenece points after bn:', reference_points.shape) # [bs, 1780, 3]
        # print('attn_mask:', attn_mask.shape) # [1780, 1780]
        # print('mask_dict:', mask_dict.keys())
        ##

        mask = x.new_zeros(x.shape[0], x.shape[2], x.shape[3])
        
        rv_pos_embeds, flow_outs_cam = self._rv_pe(x_img, x, img_metas)
        bev_pos_embeds, flow_outs_lidar = self._bev_pe(x_img, x, img_metas)
        flow_outs = flow_outs_cam if self.bev_warping == 'cam' else flow_outs_lidar
        # bev_pos_embeds = self.bev_embedding(pos2embed(self.coords_bev.to(x.device), num_pos_feats=self.hidden_dim))
        # print('rv_pose_embeds:', rv_pos_embeds.shape) #
        # print('self.coords_bev:', self.coords_bev.shape) #
        # print('bev_pose_embeds:', bev_pos_embeds.shape) #
        bev_query_embeds, rv_query_embeds = self.query_embed(reference_points, img_metas)
        query_embeds = bev_query_embeds + rv_query_embeds

        outs_dec, _ = self.transformer(
                            x, x_img, query_embeds,
                            bev_pos_embeds, rv_pos_embeds,
                            attn_masks=attn_mask
                        )
        outs_dec = torch.nan_to_num(outs_dec)

        reference = inverse_sigmoid(reference_points.clone())
        
        flag = 0
        for task_id, task in enumerate(self.task_heads, 0):
            outs = task(outs_dec)
            center = (outs['center'] + reference[None, :, :, :2]).sigmoid()
            height = (outs['height'] + reference[None, :, :, 2:3]).sigmoid()
            _center, _height = center.new_zeros(center.shape), height.new_zeros(height.shape)
            _center[..., 0:1] = center[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            _center[..., 1:2] = center[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            _height[..., 0:1] = height[..., 0:1] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            outs['center'] = _center
            outs['height'] = _height
            
            if mask_dict and mask_dict['pad_size'] > 0:
                task_mask_dict = copy.deepcopy(mask_dict)
                class_name = self.class_names[task_id]

                known_lbs_bboxes_label =  task_mask_dict['known_lbs_bboxes'][0]
                known_labels_raw = task_mask_dict['known_labels_raw']
                new_lbs_bboxes_label = known_lbs_bboxes_label.new_zeros(known_lbs_bboxes_label.shape)
                new_lbs_bboxes_label[:] = len(class_name)
                new_labels_raw = known_labels_raw.new_zeros(known_labels_raw.shape)
                new_labels_raw[:] = len(class_name)
                task_masks = [
                    torch.where(known_lbs_bboxes_label == class_name.index(i) + flag)
                    for i in class_name
                ]
                task_masks_raw = [
                    torch.where(known_labels_raw == class_name.index(i) + flag)
                    for i in class_name
                ]
                for cname, task_mask, task_mask_raw in zip(class_name, task_masks, task_masks_raw):
                    new_lbs_bboxes_label[task_mask] = class_name.index(cname)
                    new_labels_raw[task_mask_raw] = class_name.index(cname)
                task_mask_dict['known_lbs_bboxes'] = (new_lbs_bboxes_label, task_mask_dict['known_lbs_bboxes'][1])
                task_mask_dict['known_labels_raw'] = new_labels_raw
                flag += len(class_name)
                
                for key in list(outs.keys()):
                    outs['dn_' + key] = outs[key][:, :, :mask_dict['pad_size'], :]
                    outs[key] = outs[key][:, :, mask_dict['pad_size']:, :]
                outs['dn_mask_dict'] = task_mask_dict
            
            ret_dicts.append(outs)
        if self.training is False and self.vis_output is not None:
            assert isinstance(self.vis_output, dict)
            outdir = self.vis_output['outdir']
            pts_path = img_metas[0]['pts_filename']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            result_path = osp.join(outdir, file_name)
            mmcv.utils.path.mkdir_or_exist(result_path)
            for key in self.vis_output['keys'] + self.vis_output['special_keys']:
                self.vis_data.update({key: locals()[key]})
            for attr in self.vis_output['attrs']:
                self.vis_data.update({attr: getattr(self, attr)})
            self.vis_data.update(dict(lidar_file_name=file_name))
            self.vis_data.update(dict(img_metas=img_metas))
            torch.save(self.vis_data, osp.join(result_path, 'vis_data.pt'))
        return ret_dicts, flow_outs

    def forward(self, pts_feats, img_feats=None, img_metas=None):
        """
            list([bs, c, h, w])
        """
        img_metas = [img_metas for _ in range(len(pts_feats))]
        return multi_apply(self.forward_single, pts_feats, img_feats, img_metas)
    
    def _get_targets_single(self, gt_bboxes_3d, gt_labels_3d, pred_bboxes, pred_logits):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            
            gt_bboxes_3d (Tensor):  LiDARInstance3DBoxes(num_gts, 9)
            gt_labels_3d (Tensor): Ground truth class indices (num_gts, )
            pred_bboxes (list[Tensor]): num_tasks x (num_query, 10)
            pred_logits (list[Tensor]): num_tasks x (num_query, task_classes)
        Returns:
            tuple[Tensor]: a tuple containing the following.
                - labels_tasks (list[Tensor]): num_tasks x (num_query, ).
                - label_weights_tasks (list[Tensor]): num_tasks x (num_query, ).
                - bbox_targets_tasks (list[Tensor]): num_tasks x (num_query, 9).
                - bbox_weights_tasks (list[Tensor]): num_tasks x (num_query, 10).
                - pos_inds (list[Tensor]): num_tasks x Sampled positive indices.
                - neg_inds (Tensor): num_tasks x Sampled negative indices.
        """
        device = gt_labels_3d.device
        gt_bboxes_3d = torch.cat(
            (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]), dim=1
        ).to(device)
        
        task_masks = []
        flag = 0
        for class_name in self.class_names:
            task_masks.append([
                torch.where(gt_labels_3d == class_name.index(i) + flag)
                for i in class_name
            ])
            flag += len(class_name)
        
        task_boxes = []
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                task_class.append(gt_labels_3d[m] - flag2)
            task_boxes.append(torch.cat(task_box, dim=0).to(device))
            task_classes.append(torch.cat(task_class).long().to(device))
            flag2 += len(mask)
        
        def task_assign(bbox_pred, logits_pred, gt_bboxes, gt_labels, num_classes):
            num_bboxes = bbox_pred.shape[0]
            assign_results = self.assigner.assign(bbox_pred, logits_pred, gt_bboxes, gt_labels)
            sampling_result = self.sampler.sample(assign_results, bbox_pred, gt_bboxes)
            pos_inds, neg_inds = sampling_result.pos_inds, sampling_result.neg_inds
            # label targets
            labels = gt_bboxes.new_full((num_bboxes, ),
                                    num_classes,
                                    dtype=torch.long)
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            label_weights = gt_bboxes.new_ones(num_bboxes)
            # bbox_targets
            code_size = gt_bboxes.shape[1]
            bbox_targets = torch.zeros_like(bbox_pred)[..., :code_size]
            bbox_weights = torch.zeros_like(bbox_pred)
            bbox_weights[pos_inds] = 1.0
            
            if len(sampling_result.pos_gt_bboxes) > 0:
                bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
            return labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds

        labels_tasks, labels_weights_tasks, bbox_targets_tasks, bbox_weights_tasks, pos_inds_tasks, neg_inds_tasks\
             = multi_apply(task_assign, pred_bboxes, pred_logits, task_boxes, task_classes, self.num_classes)
        
        return labels_tasks, labels_weights_tasks, bbox_targets_tasks, bbox_weights_tasks, pos_inds_tasks, neg_inds_tasks
            
    def get_targets(self, gt_bboxes_3d, gt_labels_3d, preds_bboxes, preds_logits):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            gt_bboxes_3d (list[LiDARInstance3DBoxes]): batch_size * (num_gts, 9)
            gt_labels_3d (list[Tensor]): Ground truth class indices. batch_size * (num_gts, )
            pred_bboxes (list[list[Tensor]]): batch_size x num_task x [num_query, 10].
            pred_logits (list[list[Tensor]]): batch_size x num_task x [num_query, task_classes]
        Returns:
            tuple: a tuple containing the following targets.
                - task_labels_list (list(list[Tensor])): num_tasks x batch_size x (num_query, ).
                - task_labels_weight_list (list[Tensor]): num_tasks x batch_size x (num_query, )
                - task_bbox_targets_list (list[Tensor]): num_tasks x batch_size x (num_query, 9)
                - task_bbox_weights_list (list[Tensor]): num_tasks x batch_size x (num_query, 10)
                - num_total_pos_tasks (list[int]): num_tasks x Number of positive samples
                - num_total_neg_tasks (list[int]): num_tasks x Number of negative samples.
        """
        (labels_list, labels_weight_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_targets_single, gt_bboxes_3d, gt_labels_3d, preds_bboxes, preds_logits
        )
        task_num = len(labels_list[0])
        num_total_pos_tasks, num_total_neg_tasks = [], []
        task_labels_list, task_labels_weight_list, task_bbox_targets_list, \
            task_bbox_weights_list = [], [], [], []

        for task_id in range(task_num):
            num_total_pos_task = sum((inds[task_id].numel() for inds in pos_inds_list))
            num_total_neg_task = sum((inds[task_id].numel() for inds in neg_inds_list))
            num_total_pos_tasks.append(num_total_pos_task)
            num_total_neg_tasks.append(num_total_neg_task)
            task_labels_list.append([labels_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])
            task_labels_weight_list.append([labels_weight_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])
            task_bbox_targets_list.append([bbox_targets_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])
            task_bbox_weights_list.append([bbox_weights_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])
        
        return (task_labels_list, task_labels_weight_list, task_bbox_targets_list,
                task_bbox_weights_list, num_total_pos_tasks, num_total_neg_tasks)
        
    def _loss_single_task(self,
                          pred_bboxes,
                          pred_logits,
                          labels_list,
                          labels_weights_list,
                          bbox_targets_list,
                          bbox_weights_list,
                          num_total_pos,
                          num_total_neg):
        """"Compute loss for single task.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            pred_bboxes (Tensor): (batch_size, num_query, 10)
            pred_logits (Tensor): (batch_size, num_query, task_classes)
            labels_list (list[Tensor]): batch_size x (num_query, )
            labels_weights_list (list[Tensor]): batch_size x (num_query, )
            bbox_targets_list(list[Tensor]): batch_size x (num_query, 9)
            bbox_weights_list(list[Tensor]): batch_size x (num_query, 10)
            num_total_pos: int
            num_total_neg: int
        Returns:
            loss_cls
            loss_bbox 
        """
        labels = torch.cat(labels_list, dim=0)
        labels_weights = torch.cat(labels_weights_list, dim=0)
        bbox_targets = torch.cat(bbox_targets_list, dim=0)
        bbox_weights = torch.cat(bbox_weights_list, dim=0)
        
        pred_bboxes_flatten = pred_bboxes.flatten(0, 1)
        pred_logits_flatten = pred_logits.flatten(0, 1)
        
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * 0.1
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            pred_logits_flatten, labels, labels_weights, avg_factor=cls_avg_factor
        )

        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * bbox_weights.new_tensor(self.train_cfg.code_weights)[None, :]

        loss_bbox = self.loss_bbox(
            pred_bboxes_flatten[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos
        )

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox) 
        return loss_cls, loss_bbox

    def loss_single(self,
                    pred_bboxes,
                    pred_logits,
                    gt_bboxes_3d,
                    gt_labels_3d):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            pred_bboxes (list[Tensor]): num_tasks x [bs, num_query, 10].
            pred_logits (list(Tensor]): num_tasks x [bs, num_query, task_classes]
            gt_bboxes_3d (list[LiDARInstance3DBoxes]): batch_size * (num_gts, 9)
            gt_labels_list (list[Tensor]): Ground truth class indices. batch_size * (num_gts, )
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        batch_size = pred_bboxes[0].shape[0]
        pred_bboxes_list, pred_logits_list = [], []
        for idx in range(batch_size):
            pred_bboxes_list.append([task_pred_bbox[idx] for task_pred_bbox in pred_bboxes])
            pred_logits_list.append([task_pred_logits[idx] for task_pred_logits in pred_logits])
        cls_reg_targets = self.get_targets(
            gt_bboxes_3d, gt_labels_3d, pred_bboxes_list, pred_logits_list
        )
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        loss_cls_tasks, loss_bbox_tasks = multi_apply(
            self._loss_single_task, 
            pred_bboxes,
            pred_logits,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg
        )

        return sum(loss_cls_tasks), sum(loss_bbox_tasks)
    
    def _dn_loss_single_task(self,
                             pred_bboxes,
                             pred_logits,
                             mask_dict):
        known_labels, known_bboxs = mask_dict['known_lbs_bboxes']
        map_known_indice = mask_dict['map_known_indice'].long()
        known_indice = mask_dict['known_indice'].long()
        batch_idx = mask_dict['batch_idx'].long()
        bid = batch_idx[known_indice]
        known_labels_raw = mask_dict['known_labels_raw']
        
        pred_logits = pred_logits[(bid, map_known_indice)]
        pred_bboxes = pred_bboxes[(bid, map_known_indice)]
        num_tgt = known_indice.numel()

        # filter task bbox
        task_mask = known_labels_raw != pred_logits.shape[-1]
        task_mask_sum = task_mask.sum()
        
        if task_mask_sum > 0:
            # pred_logits = pred_logits[task_mask]
            # known_labels = known_labels[task_mask]
            pred_bboxes = pred_bboxes[task_mask]
            known_bboxs = known_bboxs[task_mask]

        # classification loss
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_tgt * 3.14159 / 6 * self.split * self.split  * self.split
        
        label_weights = torch.ones_like(known_labels)
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            pred_logits, known_labels.long(), label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_tgt = loss_cls.new_tensor([num_tgt])
        num_tgt = torch.clamp(reduce_mean(num_tgt), min=1).item()

        # regression L1 loss
        normalized_bbox_targets = normalize_bbox(known_bboxs, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = torch.ones_like(pred_bboxes)
        bbox_weights = bbox_weights * bbox_weights.new_tensor(self.train_cfg.code_weights)[None, :]
        # bbox_weights[:, 6:8] = 0
        loss_bbox = self.loss_bbox(
                pred_bboxes[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10], avg_factor=num_tgt)
 
        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)

        if task_mask_sum == 0:
            # loss_cls = loss_cls * 0.0
            loss_bbox = loss_bbox * 0.0

        return self.dn_weight * loss_cls, self.dn_weight * loss_bbox

    def dn_loss_single(self,
                       pred_bboxes,
                       pred_logits,
                       dn_mask_dict):
        loss_cls_tasks, loss_bbox_tasks = multi_apply(
            self._dn_loss_single_task, pred_bboxes, pred_logits, dn_mask_dict
        )
        return sum(loss_cls_tasks), sum(loss_bbox_tasks)
        
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        """"Loss function.
        Args:
            gt_bboxes_3d (list[LiDARInstance3DBoxes]): batch_size * (num_gts, 9)
            gt_labels_3d (list[Tensor]): Ground truth class indices. batch_size * (num_gts, )
            preds_dicts(tuple[list[dict]]): nb_tasks x num_lvl
                center: (num_dec, batch_size, num_query, 2)
                height: (num_dec, batch_size, num_query, 1)
                dim: (num_dec, batch_size, num_query, 3)
                rot: (num_dec, batch_size, num_query, 2)
                vel: (num_dec, batch_size, num_query, 2)
                cls_logits: (num_dec, batch_size, num_query, task_classes)
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_decoder = preds_dicts[0][0]['center'].shape[0]
        all_pred_bboxes, all_pred_logits = collections.defaultdict(list), collections.defaultdict(list)

        for task_id, preds_dict in enumerate(preds_dicts, 0):
            for dec_id in range(num_decoder):
                pred_bbox = torch.cat(
                    (preds_dict[0]['center'][dec_id], preds_dict[0]['height'][dec_id],
                    preds_dict[0]['dim'][dec_id], preds_dict[0]['rot'][dec_id],
                    preds_dict[0]['vel'][dec_id]),
                    dim=-1
                )
                all_pred_bboxes[dec_id].append(pred_bbox)
                all_pred_logits[dec_id].append(preds_dict[0]['cls_logits'][dec_id])
        all_pred_bboxes = [all_pred_bboxes[idx] for idx in range(num_decoder)]
        all_pred_logits = [all_pred_logits[idx] for idx in range(num_decoder)]

        loss_cls, loss_bbox = multi_apply(
            self.loss_single, all_pred_bboxes, all_pred_logits,
            [gt_bboxes_3d for _ in range(num_decoder)],
            [gt_labels_3d for _ in range(num_decoder)], 
        )

        loss_dict = dict()
        loss_dict['loss_cls'] = loss_cls[-1]
        loss_dict['loss_bbox'] = loss_bbox[-1]

        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(loss_cls[:-1],
                                           loss_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        
        dn_pred_bboxes, dn_pred_logits = collections.defaultdict(list), collections.defaultdict(list)
        dn_mask_dicts = collections.defaultdict(list)
        for task_id, preds_dict in enumerate(preds_dicts, 0):
            for dec_id in range(num_decoder):
                pred_bbox = torch.cat(
                    (preds_dict[0]['dn_center'][dec_id], preds_dict[0]['dn_height'][dec_id],
                    preds_dict[0]['dn_dim'][dec_id], preds_dict[0]['dn_rot'][dec_id],
                    preds_dict[0]['dn_vel'][dec_id]),
                    dim=-1
                )
                dn_pred_bboxes[dec_id].append(pred_bbox)
                dn_pred_logits[dec_id].append(preds_dict[0]['dn_cls_logits'][dec_id])
                dn_mask_dicts[dec_id].append(preds_dict[0]['dn_mask_dict'])
        dn_pred_bboxes = [dn_pred_bboxes[idx] for idx in range(num_decoder)]
        dn_pred_logits = [dn_pred_logits[idx] for idx in range(num_decoder)]
        dn_mask_dicts = [dn_mask_dicts[idx] for idx in range(num_decoder)]
        dn_loss_cls, dn_loss_bbox = multi_apply(
            self.dn_loss_single, dn_pred_bboxes, dn_pred_logits, dn_mask_dicts
        )

        loss_dict['dn_loss_cls'] = dn_loss_cls[-1]
        loss_dict['dn_loss_bbox'] = dn_loss_bbox[-1]
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(dn_loss_cls[:-1],
                                           dn_loss_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
            num_dec_layer += 1

        if self.loss_voxel_flow is not None:
            flow_output = kwargs['flow_output'][0]
            gt_flow = flow_output['gt_flow']
            pred_flow = flow_output['pred_flow']
            delta_t = flow_output['delta_t']
            flow_loss = self.loss_voxel_flow(pred_flow, gt_flow, delta_t)
            # print('flow_loss:', flow_loss)
            loss_dict.update(dict(flow_loss = flow_loss))

        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False):
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)
        
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list


@HEADS.register_module()
class CmtImageHead(CmtHead):

    def __init__(self, *args, **kwargs):
        super(CmtImageHead, self). __init__(*args, **kwargs)
        self.shared_conv = None

    def forward_single(self, x, x_img, img_metas):
        """
            x: [bs c h w]
            return List(dict(head_name: [num_dec x bs x num_query * head_dim]) ) x task_num
        """
        assert x is None
        ret_dicts = []
        
        reference_points = self.reference_points.weight
        reference_points, attn_mask, mask_dict = self.prepare_for_dn(len(img_metas), reference_points, img_metas)
        
        rv_pos_embeds, flow_outs = self._rv_pe(img_feats=x_img, img_metas=img_metas)
        
        bev_query_embeds, rv_query_embeds = self.query_embed(reference_points, img_metas)
        query_embeds = bev_query_embeds + rv_query_embeds

        outs_dec, _ = self.transformer(
                            x_img, query_embeds,
                            rv_pos_embeds,
                            attn_masks=attn_mask,
                            bs=len(img_metas)
                        )
        outs_dec = torch.nan_to_num(outs_dec)

        reference = inverse_sigmoid(reference_points.clone())
        
        flag = 0
        for task_id, task in enumerate(self.task_heads, 0):
            outs = task(outs_dec)
            center = (outs['center'] + reference[None, :, :, :2]).sigmoid()
            height = (outs['height'] + reference[None, :, :, 2:3]).sigmoid()
            _center, _height = center.new_zeros(center.shape), height.new_zeros(height.shape)
            _center[..., 0:1] = center[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            _center[..., 1:2] = center[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            _height[..., 0:1] = height[..., 0:1] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            outs['center'] = _center
            outs['height'] = _height
            
            if mask_dict and mask_dict['pad_size'] > 0:
                task_mask_dict = copy.deepcopy(mask_dict)
                class_name = self.class_names[task_id]

                known_lbs_bboxes_label =  task_mask_dict['known_lbs_bboxes'][0]
                known_labels_raw = task_mask_dict['known_labels_raw']
                new_lbs_bboxes_label = known_lbs_bboxes_label.new_zeros(known_lbs_bboxes_label.shape)
                new_lbs_bboxes_label[:] = len(class_name)
                new_labels_raw = known_labels_raw.new_zeros(known_labels_raw.shape)
                new_labels_raw[:] = len(class_name)
                task_masks = [
                    torch.where(known_lbs_bboxes_label == class_name.index(i) + flag)
                    for i in class_name
                ]
                task_masks_raw = [
                    torch.where(known_labels_raw == class_name.index(i) + flag)
                    for i in class_name
                ]
                for cname, task_mask, task_mask_raw in zip(class_name, task_masks, task_masks_raw):
                    new_lbs_bboxes_label[task_mask] = class_name.index(cname)
                    new_labels_raw[task_mask_raw] = class_name.index(cname)
                task_mask_dict['known_lbs_bboxes'] = (new_lbs_bboxes_label, task_mask_dict['known_lbs_bboxes'][1])
                task_mask_dict['known_labels_raw'] = new_labels_raw
                flag += len(class_name)
                
                for key in list(outs.keys()):
                    outs['dn_' + key] = outs[key][:, :, :mask_dict['pad_size'], :]
                    outs[key] = outs[key][:, :, mask_dict['pad_size']:, :]
                outs['dn_mask_dict'] = task_mask_dict
            
            ret_dicts.append(outs)

        return ret_dicts, flow_outs


@HEADS.register_module()
class CmtLidarHead(CmtHead):

    def __init__(self, *args, **kwargs):
        super(CmtLidarHead, self). __init__(*args, **kwargs)
        self.rv_embedding = None

    def query_embed(self, ref_points, img_metas):
        ref_points = inverse_sigmoid(ref_points.clone()).sigmoid()
        bev_embeds = self._bev_query_embed(ref_points, img_metas)
        return bev_embeds, None
    
    def forward_single(self, x, x_img, img_metas):
        """
            x: [bs c h w]
            return List(dict(head_name: [num_dec x bs x num_query * head_dim]) ) x task_num
        """
        assert x_img is None

        ret_dicts = []
        x = self.shared_conv(x)
        
        reference_points = self.reference_points.weight
        reference_points, attn_mask, mask_dict = self.prepare_for_dn(x.shape[0], reference_points, img_metas)
        
        mask = x.new_zeros(x.shape[0], x.shape[2], x.shape[3])
        
        bev_pos_embeds = self.bev_embedding(pos2embed(self.coords_bev.to(x.device), num_pos_feats=self.hidden_dim))
        bev_query_embeds, _ = self.query_embed(reference_points, img_metas)

        query_embeds = bev_query_embeds
        outs_dec, _ = self.transformer(
                            x, mask, query_embeds,
                            bev_pos_embeds,
                            attn_masks=attn_mask
                        )
        outs_dec = torch.nan_to_num(outs_dec)

        reference = inverse_sigmoid(reference_points.clone())
        
        flag = 0
        for task_id, task in enumerate(self.task_heads, 0):
            outs = task(outs_dec)
            center = (outs['center'] + reference[None, :, :, :2]).sigmoid()
            height = (outs['height'] + reference[None, :, :, 2:3]).sigmoid()
            _center, _height = center.new_zeros(center.shape), height.new_zeros(height.shape)
            _center[..., 0:1] = center[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            _center[..., 1:2] = center[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            _height[..., 0:1] = height[..., 0:1] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            outs['center'] = _center
            outs['height'] = _height
            
            if mask_dict and mask_dict['pad_size'] > 0:
                task_mask_dict = copy.deepcopy(mask_dict)
                class_name = self.class_names[task_id]

                known_lbs_bboxes_label =  task_mask_dict['known_lbs_bboxes'][0]
                known_labels_raw = task_mask_dict['known_labels_raw']
                new_lbs_bboxes_label = known_lbs_bboxes_label.new_zeros(known_lbs_bboxes_label.shape)
                new_lbs_bboxes_label[:] = len(class_name)
                new_labels_raw = known_labels_raw.new_zeros(known_labels_raw.shape)
                new_labels_raw[:] = len(class_name)
                task_masks = [
                    torch.where(known_lbs_bboxes_label == class_name.index(i) + flag)
                    for i in class_name
                ]
                task_masks_raw = [
                    torch.where(known_labels_raw == class_name.index(i) + flag)
                    for i in class_name
                ]
                for cname, task_mask, task_mask_raw in zip(class_name, task_masks, task_masks_raw):
                    new_lbs_bboxes_label[task_mask] = class_name.index(cname)
                    new_labels_raw[task_mask_raw] = class_name.index(cname)
                task_mask_dict['known_lbs_bboxes'] = (new_lbs_bboxes_label, task_mask_dict['known_lbs_bboxes'][1])
                task_mask_dict['known_labels_raw'] = new_labels_raw
                flag += len(class_name)
                
                for key in list(outs.keys()):
                    outs['dn_' + key] = outs[key][:, :, :mask_dict['pad_size'], :]
                    outs[key] = outs[key][:, :, mask_dict['pad_size']:, :]
                outs['dn_mask_dict'] = task_mask_dict
            
            ret_dicts.append(outs)

        return ret_dicts
