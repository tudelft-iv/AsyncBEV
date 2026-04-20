import torch
from torch import nn as nn
from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weighted_loss
def voxel_flow_loss(pred_flow, gt_flow, delta_t):

    ## The calculation of this loss is mainly from the DeFlow
    assert pred_flow.shape == gt_flow.shape
    speed = gt_flow.norm(dim=-1, p=2) / delta_t.squeeze(-1)

    voxelflow_loss = torch.linalg.vector_norm(pred_flow - gt_flow, dim=-1)
    # print('voxelflow_loss.shape:', voxelflow_loss.shape)
    # print('speed shape:', speed.shape)
    weight_loss = 0.0
    speed_0_4 = voxelflow_loss[speed < 0.4].mean()
    speed_mid = voxelflow_loss[(speed >= 0.4) & (speed <= 1.0)].mean()
    speed_1_0 = voxelflow_loss[speed > 1.0].mean()

    if ~speed_1_0.isnan():
        weight_loss += speed_1_0
    if ~speed_0_4.isnan():
        weight_loss += speed_0_4
    if ~speed_mid.isnan():
        weight_loss += speed_mid

    return weight_loss


@LOSSES.register_module()
class VoxelFlowLoss(nn.Module):
    """Voxel Flow Loss.

    Args:
        loss_weight (float): Weight of loss. Default: 1.0.
        reduction (str): Method to reduce losses.
            The valid reduction method are 'none', 'sum' or 'mean'.
            Default: 'mean'.
        loss_type (str): Type of loss. Options are "l1", "smooth_l1" and "mse".
            Default: "l1".
    """

    def __init__(self,
                 loss_weight=1.0):
        super(VoxelFlowLoss, self).__init__()

        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, flow_pred, flow_gt, delta_t):
        # print('In VoxelFlowLoss')
        # print('flow_pred.shape:', flow_pred.shape)
        # print('flow_gt.shape:', flow_gt.shape)
        # print('delta_t:', delta_t.shape)
        voxelflow_loss = self.loss_weight * voxel_flow_loss(pred_flow=flow_pred,
                                                            gt_flow=flow_gt,
                                                            delta_t=delta_t)
        return voxelflow_loss
