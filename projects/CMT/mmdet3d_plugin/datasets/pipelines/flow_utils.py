import os.path as osp

import torch
import numpy as np

from nuscenes.utils.geometry_utils import transform_matrix, points_in_box
from nuscenes.utils.data_classes import Box

from pyquaternion import Quaternion


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

def calculate_nusc_sceneflow(source_data,
                             target_data,
                             source_pc,
                             nusc):
    assert source_pc.shape[1] == 3, 'source_pc should be [N, 3]'
    src_data_pc = source_pc.T  # [N, 3]

    src_boxes = nusc.get_boxes(source_data['token'])
    tar_boxes = nusc.get_boxes(target_data['token'])

    ## get pose and calibaration of source data
    pose_record_src = nusc.get('ego_pose', source_data['ego_pose_token'])
    src_ego2global_translation = pose_record_src['translation']
    src_ego2global_rotation = Quaternion(pose_record_src['rotation'])
    T_src_e2g = transform_matrix(src_ego2global_translation, src_ego2global_rotation)

    cs_record_src = nusc.get('calibrated_sensor', source_data['calibrated_sensor_token'])
    src_lidar2ego_translation = cs_record_src['translation']
    src_lidar2ego_rotation = Quaternion(cs_record_src['rotation'])
    assert cs_record_src['camera_intrinsic'] == []
    T_src_l2e = transform_matrix(src_lidar2ego_translation, src_lidar2ego_rotation)
    
    ## get pose and calibaration of target data
    pose_record_tar = nusc.get('ego_pose', target_data['ego_pose_token'])
    tar_ego2global_translation = pose_record_tar['translation']
    tar_ego2global_rotation = Quaternion(pose_record_tar['rotation'])
    T_tar_e2g = transform_matrix(tar_ego2global_translation, tar_ego2global_rotation)

    cs_record_tar = nusc.get('calibrated_sensor', target_data['calibrated_sensor_token'])
    if cs_record_tar['camera_intrinsic'] == []: ## lidar data
        tar_lidar2ego_translation = cs_record_tar['translation']
        tar_lidar2ego_rotation = Quaternion(cs_record_tar['rotation'])
        T_tar_l2e = transform_matrix(tar_lidar2ego_translation, tar_lidar2ego_rotation)
    else: ## camera data
        T_tar_l2e = T_src_l2e  # use the source lidar to ego transformation for the target data
    # tar_lidar2ego_translation = cs_record_tar['translation']
    # tar_lidar2ego_rotation = Quaternion(cs_record_tar['rotation'])
    # T_tar_l2e = transform_matrix(tar_lidar2ego_translation, tar_lidar2ego_rotation)

    gt_flow_s2t = np.zeros((src_data_pc.shape[1], 3))
    T_sl2tl = np.linalg.inv(T_tar_l2e) @ np.linalg.inv(T_tar_e2g) @ T_src_e2g @ T_src_l2e

    src_data_pc_homo = np.ones((src_data_pc.shape[1], 4, 1))
    src_data_pc_homo[:, :3, 0] = src_data_pc.T[:, :3]
    target_data_pc_homo = T_sl2tl @ src_data_pc_homo
    pose_flow_s2t = target_data_pc_homo[:, :3, 0] - src_data_pc_homo[:, :3, 0]

    src_ann_instance_map = dict()
    for box in src_boxes:
        sample_annotation = nusc.get('sample_annotation', box.token)
        instance_token = sample_annotation['instance_token']
        src_ann_instance_map[box.token] = instance_token

    tar_instance_ann_map = dict()
    tar_instance_tokens_list = []
    for box in tar_boxes:
        sample_annotation = nusc.get('sample_annotation', box.token)
        instance_token = sample_annotation['instance_token']
        tar_instance_tokens_list.append(instance_token)
        tar_instance_ann_map[instance_token] = box.token

    for src_box in src_boxes:
        src_box_token = src_box.token
        src_box_translation = src_box.center
        src_box_rotation = src_box.orientation
        src_box_size = src_box.wlh

        src_box_instance_token = src_ann_instance_map[src_box_token]

        T_src_box2g = transform_matrix(src_box_translation, src_box_rotation)

        if src_box_instance_token in tar_instance_tokens_list:
            tar_box = tar_boxes[tar_instance_tokens_list.index(src_box_instance_token)]

            tar_box_translation = tar_box.center
            tar_box_rotation = tar_box.orientation

            T_tar_box2g = transform_matrix(tar_box_translation, tar_box_rotation)

            ## transform all boxes to the source lidar frame
            T_tar_box2sl = np.linalg.inv(T_src_l2e) @ np.linalg.inv(T_src_e2g) @ T_tar_box2g
            T_src_box2sl = np.linalg.inv(T_src_l2e) @ np.linalg.inv(T_src_e2g) @ T_src_box2g

            ## get transformation from from source box to target box first
            R_s2t = T_tar_box2sl @ np.linalg.inv(T_src_box2sl)

            ## get points of source lidar in the source box
            src_box2sl_translation = T_src_box2sl[:3, 3]
            src_box2sl_rotation = Quaternion(matrix=T_src_box2sl[:3, :3])
            src_box2sl = Box(src_box2sl_translation, src_box_size, src_box2sl_rotation)

            mask = points_in_box(src_box2sl, src_data_pc[:3, :])

            pts_src_box2sl = src_data_pc[:, mask]  # [4, n]

            pts_src_box2sl_homo = np.ones((pts_src_box2sl.shape[1], 4, 1))
            pts_src_box2sl_homo[:, :3, 0] = pts_src_box2sl.T[:, :3]

            pts_tar_box2sl_homo = R_s2t @ pts_src_box2sl_homo  # [N, 4, 1]

            box_flow_s2t = pts_tar_box2sl_homo[:, :3, 0] - pts_src_box2sl_homo[:, :3, 0]

            gt_flow_s2t[mask, :] = box_flow_s2t
    return gt_flow_s2t, pose_flow_s2t


def boxes2sensor(boxes, ego_pose_record, calibration_record):
    boxes_out = []

    for box in boxes:
        box2global_translation = box.center
        box2global_rotation = box.orientation
        T_box2g = transform_matrix(box2global_translation, box2global_rotation)

        ego2global_translation = ego_pose_record['translation']
        ego2global_rotation = Quaternion(ego_pose_record['rotation'])
        T_e2g = transform_matrix(ego2global_translation, ego2global_rotation)

        lidar2ego_translation = calibration_record['translation']
        lidar2ego_rotation = Quaternion(calibration_record['rotation'])
        T_l2e = transform_matrix(lidar2ego_translation, lidar2ego_rotation)

        T_box2l = np.linalg.inv(T_l2e) @ np.linalg.inv(T_e2g) @ T_box2g

        box2lidar_translation = T_box2l[:3, 3]
        box2lidar_rotation = Quaternion(matrix=T_box2l[:3, :3])

        box_out = Box(box2lidar_translation, box.wlh, box2lidar_rotation)

        boxes_out.append(box_out)

    return boxes_out