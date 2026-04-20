from typing import List
import torch.nn as nn
import torch

from mmdet3d.ops import DynamicScatter
from mmdet3d.ops import Voxelization
from mmdet3d.models import PointPillarsScatter

class FlowEmbedder(nn.Module):
    def __init__(self,
                 voxel_size: List[float] = [0.54, 0.54, 8],
                 point_cloud_range: List[float] = [-54, -54, -5, 54, 54, 3],
                 pseudo_image_dims: List[int] = [200, 200]):
        super(FlowEmbedder, self).__init__()
        self.voxelizer = Voxelization(
            max_num_points=-1,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range)
        self.cluster_scatter = DynamicScatter(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            average_points=True)
        self.pp_scatter = PointPillarsScatter(
            in_channels=3,
            output_shape=pseudo_image_dims)

    def forward(self, points_tar: List[torch.Tensor], flow_t2s: List[torch.Tensor]):
        """Forward function to scatter features."""
        voxel_info_list = []
        pseudoimage_list = []
        batch_size = len(points_tar)
        for batch_idx in range(batch_size):
            batch_points = points_tar[batch_idx]
            batch_flow = flow_t2s[batch_idx]
            assert batch_points.shape[0] == batch_flow.shape[0]
            device = batch_points.device
            ## Step1: Get non-nan points mask
            valid_point_idxes = torch.arange(batch_points.shape[0], device=device)
            not_nan_mask = ~torch.isnan(batch_points).any(dim=1)
            batch_non_nan_points = batch_points[not_nan_mask]
            batch_non_nan_flow = batch_flow[not_nan_mask]
            valid_point_idxes = valid_point_idxes[not_nan_mask]

            ## Step2: Voxelize
            batch_voxel_coords = self.voxelizer(batch_non_nan_points)
            # If any of the coords are -1, then the point is not in the voxel grid and should be discarded
            batch_voxel_coords_mask = (batch_voxel_coords != -1).all(dim=1)

            valid_batch_voxel_coords = batch_voxel_coords[batch_voxel_coords_mask]
            valid_batch_non_nan_points = batch_non_nan_points[batch_voxel_coords_mask]
            valid_point_idxes = valid_point_idxes[batch_voxel_coords_mask]
            valid_batch_non_nan_flow = batch_non_nan_flow[batch_voxel_coords_mask]

            voxel_info = dict(
                flows=valid_batch_non_nan_flow,
                points=valid_batch_non_nan_points,
                voxel_coords=valid_batch_voxel_coords,
                point_idxes=valid_point_idxes)
            ## Step3: DynamicScatter to get voxel mean and mean coors

            voxel_flow_mean, coors_flow_mean = self.cluster_scatter(valid_batch_non_nan_flow, valid_batch_voxel_coords)

            pseudoimage = self.pp_scatter(voxel_flow_mean, coors_flow_mean)[0]

            voxel_info_list.append(voxel_info)
            pseudoimage_list.append(pseudoimage)
        flow_pseudoimages = torch.cat(pseudoimage_list, dim=0)

        return voxel_info_list, flow_pseudoimages