from typing import List
import torch.nn as nn
import torch
from mmcv.runner import auto_fp16

from mmdet3d.ops import DynamicScatter
from mmdet3d.ops import Voxelization
# from projects.CMT.mmdet3d_plugin.mmcv_custom.ops.voxel import DynamicScatter, Voxelization

class FlowEmbedder(nn.Module):
    def __init__(self,
                 voxel_size: List[float] = [0.54, 0.54, 8],
                 point_cloud_range: List[float] = [-54, -54, -5, 54, 54, 3],
                 pseudo_image_dims: List[int] = [200, 200]):
        super(FlowEmbedder, self).__init__()
        self.pseudo_image_dims = pseudo_image_dims
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
            if batch_non_nan_flow.shape[0] == 0:
                voxel_info = dict(
                    flows=None,
                    points=None,
                    voxel_coords=None,
                    point_idxes=None
                )
                pseudoimage = torch.zeros(1, 3, self.pseudo_image_dims[0], self.pseudo_image_dims[0]).to(batch_non_nan_flow.device)
            else:
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


class Voxelizer(nn.Module):
    def __init__(self,
                 voxel_size: List[float] = [0.54, 0.54, 8],
                 point_cloud_range: List[float] = [-54, -54, -5, 54, 54, 3]):
        super(Voxelizer, self).__init__()
        self.voxelizer = Voxelization(
            max_num_points=-1,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range)

    def forward(self, points_tar: List[torch.Tensor]):
        """Forward function to scatter features."""
        voxel_info_list = []
        batch_size = len(points_tar)
        for batch_idx in range(batch_size):
            batch_points = points_tar[batch_idx]
            device = batch_points.device
            ## Step1: Get non-nan points mask
            valid_point_idxes = torch.arange(batch_points.shape[0], device=device)
            not_nan_mask = ~torch.isnan(batch_points).any(dim=1)
            batch_non_nan_points = batch_points[not_nan_mask]
            valid_point_idxes = valid_point_idxes[not_nan_mask]

            ## Step2: Voxelize
            batch_voxel_coords = self.voxelizer(batch_non_nan_points)
            # If any of the coords are -1, then the point is not in the voxel grid and should be discarded
            batch_voxel_coords_mask = (batch_voxel_coords != -1).all(dim=1)

            valid_batch_voxel_coords = batch_voxel_coords[batch_voxel_coords_mask]
            valid_batch_non_nan_points = batch_non_nan_points[batch_voxel_coords_mask]
            valid_point_idxes = valid_point_idxes[batch_voxel_coords_mask]

            voxel_info = dict(
                points=valid_batch_non_nan_points,
                voxel_coords=valid_batch_voxel_coords,
                point_idxes=valid_point_idxes)

            voxel_info_list.append(voxel_info)
        return voxel_info_list

class PointPillarsScatter(nn.Module):
    """Point Pillar's Scatter.

    Converts learned features from dense tensor to sparse pseudo image.

    Args:
        in_channels (int): Channels of input features.
        output_shape (list[int]): Required output shape of features.
    """

    def __init__(self, in_channels, output_shape):
        super().__init__()
        self.output_shape = output_shape
        self.ny = output_shape[0]
        self.nx = output_shape[1]
        self.in_channels = in_channels
        self.fp16_enabled = False

    @auto_fp16(apply_to=('voxel_features', ))
    def forward(self, voxel_features, coors, batch_size=None):
        """Foraward function to scatter features."""
        # TODO: rewrite the function in a batch manner
        # no need to deal with different batch cases
        if batch_size is not None:
            return self.forward_batch(voxel_features, coors, batch_size)
        else:
            return self.forward_single(voxel_features, coors)

    def forward_single(self, voxel_features, coors):
        """Scatter features of single sample.

        Args:
            voxel_features (torch.Tensor): Voxel features in shape (N, M, C).
            coors (torch.Tensor): Coordinates of each voxel.
                The first column indicates the sample ID.
        """
        # Create the canvas for this sample
        canvas = torch.zeros(
            self.in_channels,
            self.nx * self.ny,
            dtype=voxel_features.dtype,
            device=voxel_features.device)

        indices = coors[:, 1] * self.nx + coors[:, 2]
        indices = indices.long()
        voxels = voxel_features.t()
        # Now scatter the blob back to the canvas.
        canvas[:, indices] = voxels
        # Undo the column stacking to final 4-dim tensor
        canvas = canvas.view(1, self.in_channels, self.ny, self.nx)
        return [canvas]

    def forward_batch(self, voxel_features, coors, batch_size):
        """Scatter features of single sample.

        Args:
            voxel_features (torch.Tensor): Voxel features in shape (N, M, C).
            coors (torch.Tensor): Coordinates of each voxel in shape (N, 4).
                The first column indicates the sample ID.
            batch_size (int): Number of samples in the current batch.
        """
        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(
                self.in_channels,
                self.nx * self.ny,
                dtype=voxel_features.dtype,
                device=voxel_features.device)

            # Only include non-empty pillars
            batch_mask = coors[:, 0] == batch_itt
            this_coors = coors[batch_mask, :]
            indices = this_coors[:, 2] * self.nx + this_coors[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, in_channels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(batch_size, self.in_channels, self.ny,
                                         self.nx)

        return batch_canvas