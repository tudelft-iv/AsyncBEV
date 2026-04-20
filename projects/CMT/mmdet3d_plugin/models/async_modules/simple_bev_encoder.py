from idlelib.autocomplete_w import HIDE_VIRTUAL_EVENT_NAME

import torch
import torch.nn as nn
import numpy as np

from torch.nn import functional as F

from mmcv.runner import force_fp32

class SimpleBEVEncoder(nn.Module):
    def __init__(self,
                 H_bev,
                 W_bev,
                 Z_bev,
                 pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 embedding_dim=256,
                 output_dim=256):
        super(SimpleBEVEncoder, self).__init__()

        self.H_bev = H_bev
        self.Z_bev = Z_bev
        self.W_bev = W_bev
        self.pc_range = pc_range

        self.embedding_dim = embedding_dim
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=output_dim),
            nn.GELU(),
        )

    def get_bev_reference_points(self,
                                 B = 2,
                                 dtype = torch.float32,
                                 device='cuda'):
        H = self.H_bev
        W = self.W_bev
        Z = self.Z_bev

        zs = torch.linspace(0.5, Z - 0.5, Z, dtype=dtype, device=device).view(Z, 1, 1).expand(Z, H, W) / Z
        xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device).view(1, 1, W).expand(Z, H, W) / W
        ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device).view(1, H, 1).expand(Z, H, W) / H

        ref_3d = torch.stack([xs, ys, zs], dim=-1) # [8, 200, 200, 3]
        ref_3d = ref_3d.permute(0, 3, 1, 2) # [8, 3, 200, 200]
        ref_3d = ref_3d.flatten(2).permute(0,2,1) # [8, 40000, 3]
        ref_3d = ref_3d.unsqueeze(0).repeat(B, 1, 1, 1)  # Add batch dimension, [B, 8, 40000, 3]
        return ref_3d

    @force_fp32(apply_to=('reference_points_3d', 'mlvl_feats'))
    def point_sampling(self,
                       reference_points_3d,
                       img_metas):
        device = reference_points_3d.device
        dtype = reference_points_3d.dtype
        pc_range = self.pc_range

        lidar2img =[]

        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])

        lidar2img = np.asarray(lidar2img)

        lidar2img = reference_points_3d.new_tensor(lidar2img, dtype=dtype, device=device) # [B,N,4,4]

        reference_points_3d = reference_points_3d.clone() # [B, 8, 40000, 3]

        reference_points_3d [...,0:1] = reference_points_3d[...,0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points_3d [...,1:2] = reference_points_3d[...,1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points_3d [...,2:3] = reference_points_3d[...,2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]

        reference_points_3d = torch.cat([reference_points_3d, torch.ones_like(reference_points_3d[..., :1])], dim=-1)  # [B, 8, 40000, 4]

        reference_points_3d = reference_points_3d.permute(1, 0, 2, 3)  # [8, B, 40000, 4]
        D, B, num_query, _ = reference_points_3d.size()
        num_cam = lidar2img.size(1)

        reference_points_3d = reference_points_3d.view(D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)  # [8, B, num_cam, 40000, 4, 1]
        lidar2img = lidar2img.view(1, B, num_cam,1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)  # [8, B, num_cam, 40000, 4, 4]

        reference_points_cam = torch.matmul(lidar2img.to(dtype=dtype), reference_points_3d.to(dtype=dtype)).squeeze(-1) # [8, B, num_cam, 40000, 4]

        eps = torch.finfo(dtype).eps
        bev_mask = (reference_points_cam[...,2:3] > eps) # [8, B, num_cam, 40000, 1]

        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[...,2:3]) * eps)

        reference_points_cam[...,0] /= img_metas[0]['img_shape'][0][1]  # Normalize x by image width
        reference_points_cam[...,1] /= img_metas[0]['img_shape'][0][0]

        bev_mask = (
            bev_mask &
            (reference_points_cam[..., 0:1] >= 0.0) &
            (reference_points_cam[..., 0:1] <= 1.0) &
            (reference_points_cam[..., 1:2] >= 0.0) &
            (reference_points_cam[..., 1:2] <= 1.0))

        bev_mask = torch.nan_to_num(bev_mask)

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)  # [num_cam, B, 40000, 8,  2]
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1) # [num_cam, B, 40000, 8]

        return reference_points_cam, bev_mask


    def bev_projection(self,
                       reference_points_cam,
                       img_feats,
                       bev_mask):
        num_cam, B, num_query, D, _ = reference_points_cam.size()
        dtype = reference_points_cam.dtype
        device = reference_points_cam.device

        valid_index_list = []
        for i, bev_mask_per_cam in enumerate(bev_mask):
            index_query_per_cam = bev_mask_per_cam[0].sum(-1).nonzero().squeeze(-1)
            valid_index_list.append(index_query_per_cam)

        max_len = max([len(each) for each in valid_index_list])

        reference_points_rebatched = reference_points_cam.new_zeros([B, num_cam, max_len, D ,2])
        for b in range(B):
            for i, reference_points_per_cam in enumerate(reference_points_cam):
                index_query_per_cam = valid_index_list[i]
                reference_points_rebatched[b, i, :len(index_query_per_cam)] = reference_points_per_cam[b, index_query_per_cam]

        ref_points_grid_sample = reference_points_rebatched.view(B*num_cam, max_len, D, 2)
        sampling_grid = ref_points_grid_sample * 2 - 1 ## [-1, 1] range for grid_sample

        ### Assume we only use single level of features for simplicity
        B_img, num_cam_img, C, H_img, W_img = img_feats.shape
        assert B_img == B, f'Batch size must match {B_img} and {B}'
        assert num_cam_img == num_cam, f'Number of cameras must match {num_cam_img} and {num_cam}'

        img_feats_grid_sample = img_feats.view(B*num_cam, C, H_img, W_img)

        bev_feats_init = F.grid_sample(
            img_feats_grid_sample,
            sampling_grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False,
        ) # （B*num_cam, C, max_len, D)

        bev_slots =  torch.zeros([B, self.H_bev*self.W_bev, C], dtype=dtype, device=device)
        bev_feats_init = bev_feats_init.sum(-1).permute(0,2,1).view(B, num_cam, max_len, C) # (B*num_cam, C, max_len) -> (B*num_cam, max_len, C) -> (B, num_cam, max_len, C)

        for b in range(B):
            for i, index_query_per_cam in enumerate(valid_index_list):
                bev_slots[b, index_query_per_cam] += bev_feats_init[b, i, :len(index_query_per_cam)]

        valid_counts = bev_mask.sum(-1) > 0 # [num_cam, B, 40000, 8]
        valid_counts = valid_counts.permute(1, 2, 0).sum(-1) # [num_cam, B, 40000, 8] -> [B, 40000, num_cam] -> [B, 40000]
        valid_counts = torch.clamp(valid_counts, min=1.0)

        bev_slots = bev_slots / valid_counts.unsqueeze(-1) # [B, 40000, C]

        return bev_slots

    def forward(self, img_feats, img_metas):

        B = img_feats.shape[0]
        bs = int(B // 6)
        img_feats = img_feats.view(bs, 6, img_feats.shape[1], img_feats.shape[2], img_feats.shape[3]) # [B, num_cam, C, H, W]

        dtype = img_feats.dtype
        device = img_feats.device

        reference_points_3d = self.get_bev_reference_points(B=bs, dtype=dtype, device=device)
        reference_points_cam, bev_mask = self.point_sampling(reference_points_3d, img_metas)
        bev_feats = self.bev_projection(reference_points_cam, img_feats, bev_mask)

        bev_feats = self.bev_encoder(bev_feats.view(bs, self.H_bev, self.W_bev, -1).permute(0, 3, 1, 2))  # [B, C, H_bev, W_bev]

        bev_feats = bev_feats.permute(0,2,3,1).view(bs, self.H_bev*self.W_bev, -1) # [B, C, H_bev*W_bev] -> [B, H_bev*W_bev, C]
        return bev_feats ## [B, output_dim, H_bev, W_bev]

