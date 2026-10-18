"""Standalone nuScenes geometry shared by CMT and GT-flow preparation.

This module deliberately avoids importing the registered CMT plugin (and its
PyTorch/MMCV model dependencies). The original flow transforms are preserved:
scene flow describes object motion in the source LiDAR frame, while pose flow
is returned separately. Camera targets retain the source LiDAR extrinsics at
the target ego pose; points are not transformed into the camera image frame.
"""

import numpy as np

from nuscenes.utils.geometry_utils import transform_matrix, points_in_box
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion


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


def build_reference_grid(bev_size=(200, 200), resolution=0.5,
                         point_cloud_range=(-54, -54, -5, 54, 54, 3)):
    """Return the historical NumPy grid in synchronized LiDAR coordinates.

    ``bev_size`` is ``(height, width)``. The original endpoint and integer
    truncation rules are intentional: changing them changes point coordinates
    and breaks row alignment when adding coordinates to existing flow labels.
    The flattened order is z, then y, then x (x varies fastest), float64.
    """
    try:
        bev_size = np.asarray(bev_size, dtype=np.float64)
        point_cloud_range = np.asarray(point_cloud_range, dtype=np.float64)
        resolution = float(resolution)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('Grid dimensions, range, and resolution must be numeric') from error
    if bev_size.shape != (2,) or not np.isfinite(bev_size).all() or np.any(bev_size <= 0):
        raise ValueError('bev_size must contain two finite positive dimensions')
    if (point_cloud_range.shape != (6,) or
            not np.isfinite(point_cloud_range).all() or
            np.any(point_cloud_range[3:] <= point_cloud_range[:3])):
        raise ValueError('point_cloud_range must contain finite minima below maxima')
    with np.errstate(over='ignore'):
        extents = point_cloud_range[3:] - point_cloud_range[:3]
    if not np.isfinite(extents).all():
        raise ValueError('point_cloud_range extents must be finite')
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError('resolution must be finite and positive')

    h, w = bev_size
    z = point_cloud_range[5] - point_cloud_range[2]
    with np.errstate(over='ignore'):
        counts = np.array([h, w, z]) / resolution
    if not np.isfinite(counts).all() or np.any(counts < 1):
        raise ValueError('Grid dimensions divided by resolution must be at least one')
    H, W, Z = (int(value) for value in counts)

    xs = np.linspace(0.5, w - 0.5, W).reshape(1, 1, W)
    xs = np.broadcast_to(xs, (Z, H, W)) / w
    ys = np.linspace(0.5, h - 0.5, H).reshape(1, H, 1)
    ys = np.broadcast_to(ys, (Z, H, W)) / h
    zs = np.linspace(0.5, z - 0.5, Z).reshape(Z, 1, 1)
    zs = np.broadcast_to(zs, (Z, H, W)) / z
    ref_grid_3d = np.stack((xs, ys, zs), axis=-1)

    ref_grid_3d_rw = ref_grid_3d.copy()
    ref_grid_3d_rw[..., 0:1] = ref_grid_3d[..., 0:1] * (point_cloud_range[3] - point_cloud_range[0]) + point_cloud_range[0]
    ref_grid_3d_rw[..., 1:2] = ref_grid_3d[..., 1:2] * (point_cloud_range[4] - point_cloud_range[1]) + point_cloud_range[1]
    ref_grid_3d_rw[..., 2:3] = ref_grid_3d[..., 2:3] * (point_cloud_range[5] - point_cloud_range[2]) + point_cloud_range[2]
    return ref_grid_3d_rw.reshape(H * W * Z, -1)


def points_for_sample(nusc, sync_data_token, grid):
    """Select grid points in source GT-box order, preserving overlapping rows.

    Historical labels concatenate each box's selected rows independently. A
    point inside overlapping boxes therefore appears more than once. Samples
    with no boxes use one float32 zero row; boxes selecting no points can yield
    a zero-row array instead. Keep these cases distinct for label alignment.
    """
    sync_data = nusc.get('sample_data', sync_data_token)
    sync_pose_record = nusc.get('ego_pose', sync_data['ego_pose_token'])
    sync_cs_record = nusc.get('calibrated_sensor', sync_data['calibrated_sensor_token'])
    sync_boxes = nusc.get_boxes(sync_data_token)
    boxes_synclidar = boxes2sensor(sync_boxes, sync_pose_record, sync_cs_record)
    pts_in_boxes_list = []
    for box in boxes_synclidar:
        obj_mask = points_in_box(box, grid.T[:3, :])
        obj_pts = grid[obj_mask, :]
        pts_in_boxes_list.append(obj_pts)
    if len(pts_in_boxes_list) == 0:
        pts_in_boxes_list.append(np.zeros((1, 3), dtype=np.float32))
    return np.vstack(pts_in_boxes_list)
