import os.path as osp

import torch
import numpy as np

from nuscenes.utils.geometry_utils import transform_matrix, points_in_box

from pyquaternion import Quaternion

from projects.CMT.flow_geometry import boxes2sensor, calculate_nusc_sceneflow


def check_info(idx, source_frame, source_data, target_data, key, nusc, data_root):
    assert key == osp.join(data_root, target_data['filename'])
    token = target_data['token']

    # print(source_frame['sample_token'])
    assert target_data['is_key_frame'] == True
    for i in range(idx + 1):
        dt = nusc.get('sample_data', token)
        token = dt['prev']
        # print('prev_token:', token)

    # sensor2lidar_rotation', 'sensor2lidar_translation'
    assert dt['token'] == source_frame['sample_token']
    assert dt['token'] == source_data['token']
    pose_record_src = nusc.get('ego_pose', source_data['ego_pose_token'])
    cs_record_src = nusc.get('calibrated_sensor', source_data['calibrated_sensor_token'])

    assert source_frame['lidar_path'] == osp.join(data_root, source_data[
        'filename']), f"{source_frame['lidar_path']} != {osp.join(data_root, source_data['filename'])}"
    assert source_frame['timestamp'] == source_data['timestamp']
    assert source_frame['ego2global_translation'] == pose_record_src['translation']
    assert source_frame['ego2global_rotation'] == pose_record_src['rotation']

    ## get pose and calibaration of source data
    src_ego2global_translation = pose_record_src['translation']
    src_ego2global_rotation = Quaternion(pose_record_src['rotation'])
    T_src_e2g = transform_matrix(src_ego2global_translation, src_ego2global_rotation)

    src_lidar2ego_translation = cs_record_src['translation']
    src_lidar2ego_rotation = Quaternion(cs_record_src['rotation'])
    T_src_l2e = transform_matrix(src_lidar2ego_translation, src_lidar2ego_rotation)

    ## get pose and calibaration of target data
    pose_record_tar = nusc.get('ego_pose', target_data['ego_pose_token'])
    tar_ego2global_translation = pose_record_tar['translation']
    tar_ego2global_rotation = Quaternion(pose_record_tar['rotation'])
    T_tar_e2g = transform_matrix(tar_ego2global_translation, tar_ego2global_rotation)

    cs_record_tar = nusc.get('calibrated_sensor', target_data['calibrated_sensor_token'])
    tar_lidar2ego_translation = cs_record_tar['translation']
    tar_lidar2ego_rotation = Quaternion(cs_record_tar['rotation'])
    T_tar_l2e = transform_matrix(tar_lidar2ego_translation, tar_lidar2ego_rotation)

    T_sl2tl = np.linalg.inv(T_tar_l2e) @ np.linalg.inv(T_tar_e2g) @ T_src_e2g @ T_src_l2e

    sl2tl_translation = T_sl2tl[:3, 3]
    sl2tl_rotation = T_sl2tl[:3, :3]

    assert (np.allclose(source_frame['sensor2lidar_translation'],
                        sl2tl_translation)), f"{source_frame['sensor2lidar_translation']} != {sl2tl_translation}"
    assert (np.allclose(source_frame['sensor2lidar_rotation'],
                        sl2tl_rotation)), f"{source_frame['sensor2lidar_rotation']} != {sl2tl_rotation.T}"

def generate_voxel_flow(
        nusc,
        sync_data_token,
        async_data_token,
        bev_size = [200.200],
        point_cloud_range = [-54, -54, -5, 54, 54, 3],
        resolution = 0.5):
    h, w = bev_size
    z = point_cloud_range[5] - point_cloud_range[2]

    H = int(h / resolution)
    W = int(w / resolution)
    Z = int(z / resolution)

    xs = torch.linspace(0.5, w - 0.5, W).view(1, 1, W).expand(Z, H, W) / w
    ys = torch.linspace(0.5, h - 0.5, H).view(1, H, 1).expand(Z, H, W) / h
    zs = torch.linspace(0.5, z - 0.5, Z).view(-1, 1, 1).expand(Z, H, W) / z

    ref_grid_3d = torch.stack((xs, ys, zs), dim=-1)

    ref_grid_3d_rw = ref_grid_3d.clone()

    ref_grid_3d_rw[..., 0:1] = ref_grid_3d[..., 0:1] * (point_cloud_range[3]-point_cloud_range[0]) + point_cloud_range[0]
    ref_grid_3d_rw[..., 1:2] = ref_grid_3d[..., 1:2] * (point_cloud_range[4]-point_cloud_range[1]) + point_cloud_range[1]
    ref_grid_3d_rw[..., 2:3] = ref_grid_3d[..., 2:] * (point_cloud_range[5] - point_cloud_range[2]) + point_cloud_range[2]
    ref_grid_3d_flat = ref_grid_3d_rw.view(H * W * Z, -1).cpu().numpy()

    sync_data = nusc.get('sample_data', sync_data_token)
    async_data = nusc.get('sample_data', async_data_token)

    sync_pose_record = nusc.get('ego_pose', sync_data['ego_pose_token'])
    sync_cs_record = nusc.get('calibrated_sensor', sync_data['calibrated_sensor_token'])
    sync_boxes = nusc.get_boxes(sync_data_token)

    boxes_synclidar = boxes2sensor(sync_boxes, sync_pose_record, sync_cs_record)
    pts_in_boxes_list =[]
    for box in boxes_synclidar:
        obj_mask = points_in_box(box, ref_grid_3d_flat.T[:3, :])
        obj_pts = ref_grid_3d_flat[obj_mask, :]
        pts_in_boxes_list.append(obj_pts)
    pts_in_boxes_np = np.vstack(pts_in_boxes_list)

    gt_flow_sync2async = calculate_nusc_sceneflow(sync_data, async_data, pts_in_boxes_np, nusc)

    return gt_flow_sync2async
