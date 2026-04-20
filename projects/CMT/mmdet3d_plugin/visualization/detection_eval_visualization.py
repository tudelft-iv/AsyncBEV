import os, time, random

from typing import Dict, Any
import numpy as np
import mmcv

from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import DetectionMetrics, DetectionMetricDataList
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp

from projects.CMT.mmdet3d_plugin.core.eval.eval_dynamic import NuScenesEvalDynamic
from .utils import visualize_sample_custom

class DetectionEvalVis(NuScenesEvalDynamic):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def main_vis(self,
             plot_examples: int = 0,
             show_GT = True) -> Dict[str, Any]:
        """
        Main function that loads the evaluation code, visualizes samples, runs the evaluation and renders stat plots.
        :param plot_examples: How many example visualizations to write to disk.
        :param render_curves: Whether to render PR and TP curves to disk.
        :return: A dict that stores the high-level metrics and meta data.
        """
        print("In CustomVis Tool:")
        # Select a random but fixed subset to plot.
        random.seed(42)
        sample_tokens = list(self.sample_tokens)
        random.shuffle(sample_tokens)
        sample_tokens = sample_tokens[:plot_examples]

        # Visualize samples.
        example_dir = os.path.join(self.output_dir, 'examples')
        if not os.path.isdir(example_dir):
            os.mkdir(example_dir)
        for sample_token in sample_tokens:
            visualize_sample_custom(self.nusc,
                                    sample_token,
                                    self.gt_boxes if self.eval_set != 'test' else EvalBoxes(),
                                    # Don't render test GT.
                                    self.pred_boxes,
                                    eval_range=max(self.cfg.class_range.values()),
                                    show_GT = show_GT,
                                    savepath=os.path.join(example_dir, '{}.png'.format(sample_token)))

    def main_vis_per_scene(self,
                           conf_th = 0.15,
                           scene_list = None,
                           num_scenes = 0,
                           auto_conf=False,
                           with_GT = False,
                           wo_GT = False):
        print("Get Qualitative Results per scene:")
        val_split = scene_list

        scene_token = {}
        for scene in self.nusc.scene:
            scene_token[scene['name']] = scene['token']
        random.seed(42)
        random.shuffle(val_split)
        val_scene_names = val_split[:num_scenes]
        mmcv.mkdir_or_exist(self.output_dir)

        for scene_name in val_scene_names:
            scene = self.nusc.get('scene', scene_token[scene_name])
            first_sample_token = scene['first_sample_token']
            sample_token = first_sample_token
            while sample_token != '':
                w_GT_path = os.path.join(self.output_dir, 'with_GT', scene_name)
                mmcv.mkdir_or_exist(w_GT_path)
                if with_GT and not os.path.exists(os.path.join(w_GT_path,f'{sample_token}.png')):
                    visualize_sample_custom(self.nusc,
                                            sample_token,
                                            self.gt_boxes if self.eval_set != 'test' else EvalBoxes(),
                                            # Don't render test GT.
                                            self.pred_boxes,
                                            conf_th= conf_th,
                                            auto_conf=auto_conf,
                                            class_names = self.cfg.class_names,
                                            eval_range=max(self.cfg.class_range.values()),
                                            show_GT=True,
                                            savepath=os.path.join(w_GT_path,f'{sample_token}.png'))
                if wo_GT:
                    wo_GT_path = os.path.join(self.output_dir, 'without_GT', scene_name)
                    mmcv.mkdir_or_exist(wo_GT_path)
                    visualize_sample_custom(self.nusc,
                                            sample_token,
                                            self.gt_boxes if self.eval_set != 'test' else EvalBoxes(),
                                            # Don't render test GT.
                                            self.pred_boxes,
                                            conf_th= conf_th,
                                            auto_conf = auto_conf,
                                            class_names = self.cfg.class_names,
                                            eval_range=max(self.cfg.class_range.values()),
                                            show_GT=False,
                                            savepath=os.path.join(wo_GT_path,f'{sample_token}.png'))
                sample_token = self.nusc.get('sample', sample_token)['next']

    def main_vis_per_sample(self,
                            sample_token = None,
                            show_mode='all', # 'all', 'dynamic', 'static'
                            conf_th='auto', # 'auto' or a number between 0 and 1
                            with_GT=False,
                            ):
        sample = self.nusc.get('sample', sample_token)
        scene_name = self.nusc.get('scene', sample['scene_token'])['name']
        if with_GT:
            path = os.path.join(self.output_dir, 'with_GT', scene_name)
        else:
            path = os.path.join(self.output_dir, 'without_GT', scene_name)
        mmcv.mkdir_or_exist(path)
        fig = visualize_sample_custom(self.nusc,
                                sample_token,
                                self.gt_boxes if self.eval_set != 'test' else EvalBoxes(),
                                # Don't render test GT.
                                self.pred_boxes,
                                conf_th = conf_th,
                                class_names=self.cfg.class_names,
                                eval_range=max(self.cfg.class_range.values()),
                                show_GT=with_GT,
                                show_mode=show_mode,
                                savepath=os.path.join(path, f'{sample_token}.png'))
        return fig

    def evaluate_per_sample(self,
                            mode,
                            sample_token=None):

        """
        Performs the actual evaluation for dynamic and static objects separately.
        :return: A tuple of high-level and the raw metric data.
        """
        start_time = time.time()

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        if self.verbose:
            print('Accumulating metric data...')
        metric_data_list = DetectionMetricDataList()

        gt_boxes_per_sample = EvalBoxes()
        pred_boxes_per_sample = EvalBoxes()

        if mode == 'all':
            gt_boxes = self.gt_boxes[sample_token]
            pred_boxes = self.pred_boxes[sample_token]

        elif mode == 'dynamic':
            gt_boxes = self.gt_dynamic_boxes[sample_token]
            pred_boxes = self.pred_dynamic_boxes[sample_token]

        elif mode == 'static':
            gt_boxes = self.gt_static_boxes[sample_token]
            pred_boxes = self.pred_static_boxes[sample_token]
        else:
            raise ValueError(f'Error: Invalid mode {mode}!')

        gt_boxes_per_sample.add_boxes(sample_token, gt_boxes)
        pred_boxes_per_sample.add_boxes(sample_token, pred_boxes)

        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                md = accumulate(gt_boxes_per_sample, pred_boxes_per_sample, class_name, self.cfg.dist_fcn_callable, dist_th)
                metric_data_list.set(class_name, dist_th, md)

        # -----------------------------------
        # Step 2: Calculate metrics from the data.
        # -----------------------------------
        if self.verbose:
            print('Calculating metrics...')
        metrics = DetectionMetrics(self.cfg)
        for class_name in self.cfg.class_names:
            # Compute APs.
            for dist_th in self.cfg.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.cfg.min_recall, self.cfg.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)

            # Compute TP metrics.
            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(class_name, self.cfg.dist_th_tp)]
                if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                    tp = np.nan
                elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                    tp = np.nan
                else:
                    tp = calc_tp(metric_data, self.cfg.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        # Compute evaluation time.
        metrics.add_runtime(time.time() - start_time)

        return metrics, metric_data_list





