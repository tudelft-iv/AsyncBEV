## Including the data load pipeline for the asynchronous data loading

# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import os.path as osp
import numpy as np
import torch
from mmdet3d.core.points import get_points_type
from mmdet.datasets.builder import PIPELINES
from mmdet3d.datasets.pipelines import LoadPointsFromFile, LoadPointsFromMultiSweeps
import pickle

@PIPELINES.register_module()
class LoadMultiViewAsyncImageFromFiles(object):
    """Load multi channel asynchronous images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
        async_frames (int): Number of frames to load asynchronously. Defaults to 1, max 6.
        map_file (str): The map file for asynchronous loading.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='unchanged',
                 async_frames=0,
                 random_sampling=False,
                 map_file=None):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.async_frames = async_frames
        self.random_sampling = random_sampling
        self.map_file = map_file
        assert map_file is not None, "map_file must be provided for async loading"
        self.sample_sweep_map = mmcv.load(map_file)

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """

        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        async_img_list = []
        async_img_metas_list = []
        if self.random_sampling:
            self.async_frames = torch.randint(0, 9, (1,)).item()
        # 8 is the maximum number of frames between two key frames
        for name in filename:
            idx = min(self.async_frames, len(self.sample_sweep_map[name]) - 1)

            async_img_dict = self.sample_sweep_map[name][idx]
            # print(async_img_dict.keys())
            async_img_name = async_img_dict['data_path']
            async_img_list.append(mmcv.imread(async_img_name,self.color_type))
            async_img_metas_list.append(dict(
                ego2global_translation = async_img_dict['ego2global_translation'],
                ego2global_rotation = async_img_dict['ego2global_rotation'],
                sample_file_name = name,
                timestamp = async_img_dict['timestamp'] / 1e6,
                async_data_token = async_img_dict['data_token'],
                async_img_name = async_img_name,
                key_pose = True if 'BACK_LEFT' in async_img_name else False
            ))

        img = np.stack(async_img_list, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['async_metas'] = async_img_metas_list
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        repr_str += f"async_frames={self.async_frames})"
        repr_str += f"map_file={self.map_file})"
        return repr_str


@PIPELINES.register_module()
class LoadAsyncPointsFromFile(LoadPointsFromFile):
    """Load Async points from a sammple point file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.

        load_dim (int): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        async_frames (int):
            - Number of frames to load asynchronously before the key frame.
            - Defaults to 1, max 10.
        map_file (str): The map file for asynchronous loading.
    """

    def __init__(self,
                 coord_type='LIDAR',
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False,
                 file_client_args=dict(backend='disk'),
                 async_frames=1,
                 random_sampling=False,
                 map_file=None):
        super().__init__(
            coord_type=coord_type,
            load_dim=load_dim,
            use_dim=use_dim,
            shift_height=shift_height,
            use_color=use_color,
            file_client_args=file_client_args)

        self.async_frames = async_frames
        self.map_file = map_file
        self.random_sampling = random_sampling

        assert map_file is not None, "map_file must be provided for async loading"
        self.async_sample_sweep_map = mmcv.load(map_file)

    def __call__(self, results):
        """Call function to load async points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """

        pts_filename = results['pts_filename']
        # ts = results['timestamp']

        async_sample_sweep = self.async_sample_sweep_map[pts_filename]
        if self.random_sampling:
            self.async_frames = torch.randint(0, 13, (1,)).item()

        idx = min(self.async_frames, len(async_sample_sweep) - 1)

        aync_pts_filename = async_sample_sweep[idx]['lidar_path']

        points = self._load_points(aync_pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]

        # sweep_ts = async_sample_sweep[idx]['timestamp'] / 1e6
        # points[:, :3]  = points[:, :3] @ async_sample_sweep[idx]['sensor2lidar_rotation'].T
        # points[:, :3] = points[:, :3] + async_sample_sweep[idx]['sensor2lidar_translation']
        # points[:, 4] = ts - sweep_ts

        prev_sweeps = async_sample_sweep[idx]['prev_sweeps']

        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points

        results['async_prev_sweeps'] = prev_sweeps
        results['async_metas'] = dict(
            async_data_token = async_sample_sweep[idx]['sample_token'],
            timestamp = async_sample_sweep[idx]['timestamp'] / 1e6,
            sensor2lidar_rotation = async_sample_sweep[idx]['sensor2lidar_rotation'],
            sensor2lidar_translation = async_sample_sweep[idx]['sensor2lidar_translation'],
            ego2global_rotation = async_sample_sweep[idx]['ego2global_rotation'],
            ego2global_translation = async_sample_sweep[idx]['ego2global_translation'],
        )
        return results


@PIPELINES.register_module()
class LoadAsyncPointsFromMultiSweeps(LoadPointsFromMultiSweeps):
    """Load points from multiple sweeps for the asynchronized sample.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """
    def __init__(self,
                 sweeps_num=10,
                 load_dim=5,
                 use_dim=[0, 1, 2, 4],
                 file_client_args=dict(backend='disk'),
                 pad_empty_sweeps=False,
                 remove_close=False,
                 early_motion_compensation = False,
                 test_mode=False):
        super().__init__(
            sweeps_num=sweeps_num,
            load_dim=load_dim,
            use_dim=use_dim,
            file_client_args=file_client_args,
            pad_empty_sweeps=pad_empty_sweeps,
            remove_close=remove_close,
            test_mode=test_mode)
        self.early_motion_compensation = early_motion_compensation

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        points = results['points']
        points.tensor[:, 4] = 0
        async_prev_sweeps = results['async_prev_sweeps']
        ## the 4th dimension is the relative timestamp, already modified in the LoadPointsFromFile function
        sweep_points_list = [points]
        ts0 = results['timestamp']
        ts = results['async_metas']['timestamp']

        if self.pad_empty_sweeps and len(async_prev_sweeps) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(async_prev_sweeps) <= self.sweeps_num:
                choices = np.arange(len(async_prev_sweeps))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                choices = np.random.choice(len(async_prev_sweeps), self.sweeps_num, replace=False)
            for idx in choices:
                sweep = async_prev_sweeps[idx]
                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                ## transfomr from sweeps to the async frame
                sweep_ts = sweep['timestamp'] / 1e6
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)

        if self.early_motion_compensation:
        ## transfomr points from the async frame to the key frame (sample frame)
            points.tensor[:, :3] = points.tensor[:,:3] @ results['async_metas']['sensor2lidar_rotation'].T
            points.tensor[:, :3] += results['async_metas']['sensor2lidar_translation']
            points.tensor[:, 4] += ts0 - ts

        points = points[:, self.use_dim]
        results['points'] = points

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(sweeps_num={self.sweeps_num}, '
        repr_str += f"load_dim={self.load_dim}, "
        repr_str += f"use_dim={self.use_dim}, "
        repr_str += f"pad_empty_sweeps={self.pad_empty_sweeps}, "
        repr_str += f"remove_close={self.remove_close}, "
        repr_str += f"test_mode={self.test_mode})"
        return repr_str

@PIPELINES.register_module()
class LoadGTVoxelFlow(object):
    def __init__(self,
                 flow_folder='data/nuscenes/async_supplement/async_lidar_flow'):
        self.flow_folder = flow_folder

    def __call__(self, results):
        if 'lidar' in self.flow_folder:
            async_data_token = results['async_metas']['async_data_token']
        elif 'cam' in self.flow_folder:
            async_data_token = results['async_metas'][4]['async_data_token'] ## Using CAM_BACK_LEFT
            assert results['async_metas'][4]['key_pose']
        else:
            raise ValueError(f'flow_folder should contain lidar or cam, but got {self.flow_folder}.')

        sample_token = results['sample_idx']
        preprocessed_flow_path = osp.join(self.flow_folder, sample_token, f'{async_data_token}.pkl')
        with open(preprocessed_flow_path, 'rb') as f:
            preprocessed_flow = pickle.load(f)

        pts_in_boxes_sync = preprocessed_flow['pts_in_boxes']
        scene_flow_sync2async = preprocessed_flow['gt_flow_sync2async']
        if 'pose_flow_sync2async' in preprocessed_flow.keys():
            pose_flow_sync2async = preprocessed_flow['pose_flow_sync2async']
        elif 'pose_flow_s2t' in preprocessed_flow.keys():
            pose_flow_sync2async = preprocessed_flow['pose_flow_s2t']
        else:
            raise ValueError(f'No pose flow found in the preprocessed flow file.{preprocessed_flow.keys()}, sample_token:{sample_token}, async_data_token: {async_data_token}.')

        pts_in_boxes_async = pts_in_boxes_sync + scene_flow_sync2async + pose_flow_sync2async
        scene_flow_async2sync = - scene_flow_sync2async
        pose_flow_async2sync = - pose_flow_sync2async


        results['gt_flow_sync2async'] = scene_flow_sync2async
        results['pts_in_boxes_sync'] = pts_in_boxes_sync
        results['gt_flow_async2sync'] = scene_flow_async2sync
        results['pts_in_boxes_async'] = pts_in_boxes_async
        # results['pose_flow_s2t'] = preprocessed_flow['pose_flow_sync2async']

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(bev_size={self.bev_size}, '
        repr_str += f"point_cloud_range={self.point_cloud_range}, "
        repr_str += f"resolution={self.resolution})"
        return repr_str

# @PIPELINES.register_module()
# class LoadGTVoxelFlow(object):
#     def __init__(self,
#                  bev_size=[200, 200],
#                  point_cloud_range=[-54, -54, -5, 54, 54, 3],
#                  resolution=0.5,
#                  nusc_version='v1.0-trainval',
#                  data_root='data/nuscenes'):
#         self.nusc = NuScenes(version=nusc_version, dataroot=data_root, verbose=False)
#         self.bev_size = bev_size
#         self.point_cloud_range = point_cloud_range
#         self.resolution = resolution
#
#         h, w = bev_size # [200, 200]
#         z = point_cloud_range[5] - point_cloud_range[2]
#
#         H = int(h / resolution) # 400
#         W = int(w / resolution) # 400
#         Z = int(z / resolution) # 400
#
#         xs = np.linspace(0.5, w - 0.5, W).reshape(1, 1, W)
#         xs = np.broadcast_to(xs, (Z, H, W)) / w
#
#         ys = np.linspace(0.5, h - 0.5, H).reshape(1, H, 1)
#         ys = np.broadcast_to(ys, (Z, H, W)) / h
#
#         zs = np.linspace(0.5, z - 0.5, Z).reshape(Z, 1, 1)
#         zs = np.broadcast_to(zs, (Z, H, W)) / z
#
#         ref_grid_3d = np.stack((xs, ys, zs), axis=-1)
#
#         ref_grid_3d_rw = ref_grid_3d.copy()
#         ref_grid_3d_rw[..., 0:1] = ref_grid_3d[..., 0:1] * (point_cloud_range[3] - point_cloud_range[0]) + point_cloud_range[0]
#         ref_grid_3d_rw[..., 1:2] = ref_grid_3d[..., 1:2] * (point_cloud_range[4] - point_cloud_range[1]) + point_cloud_range[1]
#         ref_grid_3d_rw[..., 2:3] = ref_grid_3d[..., 2:3] * (point_cloud_range[5] - point_cloud_range[2]) + point_cloud_range[2]
#
#         self.ref_grid_3d_flat = ref_grid_3d_rw.reshape(H * W * Z, -1)
#
#     def __call__(self, results):
#         async_data_token = results['async_metas']['async_data_token']
#         sample_token = results['sample_idx']
#         sample_rec = self.nusc.get('sample', sample_token)
#         sync_data_token = sample_rec['data']['LIDAR_TOP']
#
#         sync_data = self.nusc.get('sample_data', sync_data_token)
#         async_data = self.nusc.get('sample_data', async_data_token)
#
#         sync_pose_record = self.nusc.get('ego_pose', sync_data['ego_pose_token'])
#         sync_cs_record = self.nusc.get('calibrated_sensor', sync_data['calibrated_sensor_token'])
#         sync_boxes = self.nusc.get_boxes(sync_data_token)
#
#         boxes_synclidar = boxes2sensor(sync_boxes, sync_pose_record, sync_cs_record)
#         pts_in_boxes_list = []
#         for box in boxes_synclidar:
#             obj_mask = points_in_box(box, self.ref_grid_3d_flat.T[:3, :])
#             obj_pts = self.ref_grid_3d_flat[obj_mask, :]
#             pts_in_boxes_list.append(obj_pts)
#         if len(pts_in_boxes_list) == 0:
#             pts_in_boxes_list.append(np.zeros((1, 3), dtype=np.float32))
#
#         pts_in_boxes_np = np.vstack(pts_in_boxes_list)
#
#         gt_flow_sync2async, pose_flow_s2t = calculate_nusc_sceneflow(sync_data, async_data, pts_in_boxes_np, self.nusc)
#
#         results['gt_flow_sync2async'] = gt_flow_sync2async
#         results['pts_in_boxes'] = pts_in_boxes_np
#         results['pose_flow_s2t'] = pose_flow_s2t
#
#         return results
#
#     def __repr__(self):
#         """str: Return a string that describes the module."""
#         repr_str = self.__class__.__name__
#         repr_str += f'(bev_size={self.bev_size}, '
#         repr_str += f"point_cloud_range={self.point_cloud_range}, "
#         repr_str += f"resolution={self.resolution})"
#         return repr_str










