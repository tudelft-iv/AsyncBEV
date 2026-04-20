plugin=True
plugin_dir='projects/CMT/mmdet3d_plugin/'

exp_id = 'cmt_voxel0075_vov_1600x640_cbgs'
samples_per_gpu = 4
workers_per_gpu = 8

ckpg_save_interval = 2
eval_interval = 2
data_root = 'data/nuscenes/'
sub_dir = 'mmdet3d_cmt/'
train_ann_file = sub_dir + 'nuscenes_infos_train.pkl'
val_ann_file = sub_dir + 'nuscenes_infos_val.pkl'

if 'mini' in train_ann_file:
    work_dir = f'./outputs/code_test/{exp_id}/'
    nusc_version = 'v1.0-mini'
else:
    work_dir = f'./outputs/train/{exp_id}/'
    nusc_version = 'v1.0-trainval'
file_client_args = dict(backend='disk')
eval_async_frame = 9
bev_warping_modality = 'lidar'
async_file_map_path = 'async_supplement/lidar_sample_sweeps_map_w_extrin_ego_sample_ref_coord.pkl'


point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
voxel_size = [0.075, 0.075, 0.2]
out_size_factor = 8
evaluation = dict(interval=eval_interval)
dataset_type = 'CustomNuScenesDataset'
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False)
    
ida_aug_conf = {
        "resize_lim": (0.94, 1.25),
        "final_dim": (640, 1600),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 900,
        "W": 1600,
        "rand_flip": True,
    }

test_pipeline = [
    dict(
        type='LoadAsyncPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        async_frames=eval_async_frame,
        map_file = data_root + async_file_map_path,
    ),
    dict(
        type='LoadAsyncPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True
    ),
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadGTVoxelFlow',
        flow_folder=data_root + 'async_supplement/async_lidar_flow',
    ),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=False),
            dict(type='NormalizeMultiviewImage', **img_norm_cfg),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D',
                 keys=['points', 'img'],
                 meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                            'depth2img', 'cam2img', 'pad_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'pcd_trans', 'sample_idx',
                            'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                            'transformation_3d_flow', 'rot_degree',
                            'gt_bboxes_3d', 'gt_labels_3d',
                            'ego2global_translation',
                            'ego2global_rotation',
                            'lidar2ego_translation',
                            'lidar2ego_rotation',
                            'async_metas',
                            'timestamp',
                            'gt_flow_sync2async',
                            'gt_flow_async2sync',
                            'pts_in_boxes_sync',
                            'pts_in_boxes_async',
                            'pose_flow_s2t')
                 )
        ])
]
data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=workers_per_gpu,
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + val_ann_file,
        load_interval=1,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + val_ann_file,
        load_interval=1,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'))
model = dict(
    type='CmtDetector',
    use_grid_mask=True,
    img_backbone=dict(
        type='VoVNet',
        spec_name='V-99-eSE',
        norm_eval=True,
        frozen_stages=-1,
        input_ch=3,
        out_features=('stage4','stage5',)),
    img_neck=dict(
        type='CPFPN',
        in_channels=[768, 1024],
        out_channels=256,
        num_outs=2),
    pts_voxel_layer=dict(
        num_point_features=5,
        max_num_points=10,
        voxel_size=voxel_size,
        max_voxels=(120000, 160000),
        point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(
        type='HardSimpleVFE',
        num_features=5,
    ),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=5,
        sparse_shape=[41, 1440, 1440],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='CmtHead',
        in_channels=512,
        hidden_dim=256,
        downsample_scale=8,
        async_warp_modality=bev_warping_modality,
        common_heads=dict(center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        tasks=[
            dict(num_class=10, class_names=[
                'car', 'truck', 'construction_vehicle',
                'bus', 'trailer', 'barrier',
                'motorcycle', 'bicycle',
                'pedestrian', 'traffic_cone'
            ]),
        ],
        bbox_coder=dict(
            type='MultiTaskBBoxCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10), 
        separate_head=dict(
            type='SeparateTaskHead', init_bias=-2.19, final_kernel=1),
        transformer=dict(
            type='CmtTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    with_cp=False,
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadFlashAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),

                    feedforward_channels=1024, #unused
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2, alpha=0.25, reduction='mean', loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=1.0),
    ),
    train_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            assigner=dict(
                type='HungarianAssigner3D',
                # cls_cost=dict(type='ClassificationCost', weight=2.0),
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head. 
                pc_range=point_cloud_range,
                code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            ),
            pos_weight=-1,
            gaussian_overlap=0.1,
            min_radius=2,
            grid_size=[1440, 1440, 40],  # [x_len, y_len, 1]
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            point_cloud_range=point_cloud_range)),
    test_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            grid_size=[1440, 1440, 40],
            out_size_factor=out_size_factor,
            pc_range=point_cloud_range,
            voxel_size=voxel_size,
            nms_type=None,
            nms_thr=0.2,
            use_rotate_nms=True,
            max_num=200
        )))
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.01, decay_mult=5),
            'img_neck': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)  # for 8gpu * 2sample_per_gpu
optimizer_config = dict(
    type='CustomFp16OptimizerHook',
    loss_scale='dynamic',
    grad_clip=dict(max_norm=35, norm_type=2),
    custom_fp16=dict(pts_voxel_encoder=False, pts_middle_encoder=False, pts_bbox_head=False))
lr_config = dict(
    policy='cyclic',
    target_ratio=(8, 0.0001),
    cyclic_times=1,
    step_ratio_up=0.4)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.4)
total_epochs = 20
checkpoint_config = dict(interval=ckpg_save_interval)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook'),
           dict(type='WandbLoggerHook',
                init_kwargs=dict(
                    project='cmt_async',
                    name=f'{exp_id}'))
           ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
custom_hooks = [
    dict(type='CheckpointLateStageHook',
         start=1,
         priority=60)
]

resume_from = None #'/home/shimingwang/workspace/mmdet3d_v100rc5/outputs/train/cmt_voxel0075_vov_1600x640_cbgs/epoch_8.pth'
workflow = [('train', 1)]
# gpu_ids = range(0, 8)
