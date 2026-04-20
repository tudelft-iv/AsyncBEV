
import os
import time
import json
from typing import Tuple, Dict, Any
import numpy as np
from prettytable import PrettyTable
import tqdm
from nuscenes import NuScenes

from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import add_center_dist, filter_eval_boxes

from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionMetricDataList, DetectionBox
from nuscenes.eval.detection.utils import category_to_detection_name

from nuscenes.utils.splits import create_splits_scenes


class NuScenesEvalDynamic:
    def __init__(self,
                 nusc: NuScenes,
                 config: DetectionConfig,
                 result_path: str,
                 eval_set: str,
                 output_dir: str = None,
                 verbose: bool = True):
        super().__init__()
        """
        Initialize a DetectionEval object.
        :param nusc: A NuScenes object.
        :param config: A DetectionConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        """

        self.nusc = nusc
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config

        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        # Load predictions, GT boxes, dynamic GT Boxes, and static GT boxes.
        if verbose:
            print('Initializing nuScenes detection evaluation')
        self.pred_boxes, self.pred_dynamic_boxes, self.pred_static_boxes, self.meta = load_dynamic_prediction(self.result_path, self.cfg.max_boxes_per_sample, DetectionBox, verbose=verbose)
        self.gt_boxes, self.gt_dynamic_boxes, self.gt_static_boxes = load_dynamic_gt(self.nusc, self.eval_set, DetectionBox, verbose)
        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), "Samples in split doesn't match samples in predictions."

        # GT boxes preprocess and filter
        self.pred_boxes = add_center_dist(nusc, self.pred_boxes)
        self.pred_dynamic_boxes = add_center_dist(nusc, self.pred_dynamic_boxes)
        self.pred_static_boxes = add_center_dist(nusc, self.pred_static_boxes)

        self.gt_boxes = add_center_dist(nusc, self.gt_boxes)
        self.gt_dynamic_boxes = add_center_dist(nusc, self.gt_dynamic_boxes)
        self.gt_static_boxes = add_center_dist(nusc, self.gt_static_boxes)

        if verbose:
            print('Filtering predictions')
        self.pred_boxes = filter_eval_boxes(nusc, self.pred_boxes, self.cfg.class_range, verbose=verbose)
        print()
        if verbose:
            print('Filtering dynamic predictions')
        self.pred_dynamic_boxes = filter_eval_boxes(nusc, self.pred_dynamic_boxes, self.cfg.class_range, verbose=verbose)
        print()
        if verbose:
            print('Filtering static predictions')
        self.pred_static_boxes = filter_eval_boxes(nusc, self.pred_static_boxes, self.cfg.class_range, verbose=verbose)
        print()
        if verbose:
            print('Filtering all ground truth annotations.')
        self.gt_boxes = filter_eval_boxes(nusc, self.gt_boxes, self.cfg.class_range, verbose=verbose)
        print()
        if verbose:
            print('Filtering dynamic ground truth boxes')
        self.gt_dynamic_boxes = filter_eval_boxes(nusc, self.gt_dynamic_boxes, self.cfg.class_range, verbose=verbose)
        print()
        if verbose:
            print('Filtering static ground truth boxes')
        self.gt_static_boxes = filter_eval_boxes(nusc, self.gt_static_boxes, self.cfg.class_range, verbose=verbose)
        print()

        self.sample_tokens = self.gt_boxes.sample_tokens

    def evaluate(self, mode) -> Tuple[DetectionMetrics, DetectionMetricDataList]:
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

        if mode == 'all':
            gt_boxes = self.gt_boxes
            pred_boxes = self.pred_boxes

        elif mode == 'dynamic':
            gt_boxes = self.gt_dynamic_boxes
            pred_boxes = self.pred_dynamic_boxes

        elif mode == 'static':
            gt_boxes = self.gt_static_boxes
            pred_boxes = self.pred_static_boxes
        else:
            raise ValueError(f'Error: Invalid mode {mode}!')

        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                md = accumulate(gt_boxes, pred_boxes, class_name, self.cfg.dist_fcn_callable, dist_th)
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

    def evaluate_per_sample(self, sample_token, mode) -> Tuple[DetectionMetrics, DetectionMetricDataList]:
        """
        Performs the actual evaluation for each sample for dynamic and static objects separately.
        :return: A tuple of high-level and the raw metric data.
        """
        start_time = time.time()

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        if self.verbose:
            print('Accumulating metric data...')
        metric_data_list = DetectionMetricDataList()

        if mode == 'all':
            gt_boxes = EvalBoxes()
            gt_boxes.add_boxes(sample_token, self.gt_boxes.boxes[sample_token])
            pred_boxes = EvalBoxes()
            pred_boxes.add_boxes(sample_token, self.pred_boxes.boxes[sample_token])

        elif mode == 'dynamic':
            gt_boxes = EvalBoxes()
            gt_boxes.add_boxes(sample_token, self.gt_dynamic_boxes.boxes[sample_token])
            pred_boxes = EvalBoxes()
            pred_boxes.add_boxes(sample_token, self.pred_dynamic_boxes.boxes[sample_token])

        elif mode == 'static':
            gt_boxes = EvalBoxes()
            gt_boxes.add_boxes(sample_token, self.gt_static_boxes.boxes[sample_token])
            pred_boxes = EvalBoxes()
            pred_boxes.add_boxes(sample_token, self.pred_static_boxes.boxes[sample_token])
        else:
            raise ValueError(f'Error: Invalid mode {mode}!')

        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                md = accumulate(gt_boxes, pred_boxes, class_name, self.cfg.dist_fcn_callable, dist_th)
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

    def main(self) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
                Main function that loads the evaluation code, visualizes samples, runs the evaluation and renders stat plots.
                :param plot_examples: How many example visualizations to write to disk.
                :param render_curves: Whether to render PR and TP curves to disk.
                :return: A dict that stores the high-level metrics and meta data.
                """

        # Run evaluation.
        metrics_all, metric_data_list_all = self.evaluate(mode='all')
        metrics_dym, metric_data_list_dym = self.evaluate(mode='dynamic')
        metrics_stc, metric_data_list_stc = self.evaluate(mode='static')

        # Print and save logs.
        print('##'*10)
        print('Evaluation of All objects:')
        metrics_summary_all = self.printing_logs(metrics_all, metric_data_list_all, 'all')
        print('##'*10)
        print('Evaluation of Dynamic objects:')
        metrics_summary_dym = self.printing_logs(metrics_dym, metric_data_list_dym, 'dynamic')
        print('##'*10)
        print('Evaluation of Static objects:')
        metrics_summary_stc = self.printing_logs(metrics_stc, metric_data_list_stc, 'static')
        print('##'*10)

        return metrics_summary_all, metrics_summary_dym, metrics_summary_stc

    def printing_logs(self, metrics, metric_data_list, mode) -> Dict[str, Any]:
        assert mode in ['all', 'dynamic', 'static'], f'Error: Invalid mode: {mode}!'
        # Dump the metric data, meta and metrics to disk.
        if self.verbose:
            print(f'Saving metrics to: {self.output_dir}')
        metrics_summary = metrics.serialize()
        metrics_summary['meta'] = self.meta.copy()
        with open(os.path.join(self.output_dir, f'metrics_summary_{mode}.json'), 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        with open(os.path.join(self.output_dir, f'metrics_details_{mode}.json'), 'w') as f:
            json.dump(metric_data_list.serialize(), f, indent=2)

        # Print high-level metrics.
        print('mAP: %.4f' % (metrics_summary['mean_ap']))
        err_name_mapping = {
            'trans_err': 'mATE',
            'scale_err': 'mASE',
            'orient_err': 'mAOE',
            'vel_err': 'mAVE',
            'attr_err': 'mAAE'
        }
        for tp_name, tp_val in metrics_summary['tp_errors'].items():
            print('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
        print('NDS: %.4f' % (metrics_summary['nd_score']))
        print('Eval time: %.1fs' % metrics_summary['eval_time'])


        # Print per-class metrics.
        print()
        print('Per-class results:')
        eval_results_table = PrettyTable()

        eval_results_table.field_names = ['Object Class', 'AP', 'ATE', 'ASE', 'AOE', 'AVE', 'AAE']
        class_aps = metrics_summary['mean_dist_aps']
        class_tps = metrics_summary['label_tp_errors']
        for class_name in class_aps.keys():
            eval_results_table.add_row([
                class_name,
                f"{class_aps[class_name]:.3f}",
                f"{class_tps[class_name]['trans_err']:.3f}",
                f"{class_tps[class_name]['scale_err']:.3f}",
                f"{class_tps[class_name]['orient_err']:.3f}",
                f"{class_tps[class_name]['vel_err']:.3f}",
                f"{class_tps[class_name]['attr_err']:.3f}"])
        print(eval_results_table)
        # print()
        # print('Per-class results:')
        # print('Object Class\tAP\tATE\tASE\tAOE\tAVE\tAAE')
        # class_aps = metrics_summary['mean_dist_aps']
        # class_tps = metrics_summary['label_tp_errors']
        # for class_name in class_aps.keys():
        #     print('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
        #           % (class_name, class_aps[class_name],
        #              class_tps[class_name]['trans_err'],
        #              class_tps[class_name]['scale_err'],
        #              class_tps[class_name]['orient_err'],
        #              class_tps[class_name]['vel_err'],
        #              class_tps[class_name]['attr_err']))
        return metrics_summary

def load_dynamic_gt(nusc: NuScenes, eval_split: str, box_cls, verbose: bool = False) -> Tuple[EvalBoxes, EvalBoxes, EvalBoxes]:
    """
    Loads ground truth boxes from DB.
    :param nusc: A NuScenes instance.
    :param eval_split: The evaluation split for which we load GT boxes.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The GT boxes.
    """
    # Init.
    if box_cls == DetectionBox:
        attribute_map = {a['token']: a['name'] for a in nusc.attribute}
    else:
        raise NotImplementedError('Error: Invalid box_cls %s!' % box_cls)
    if verbose:
        print('Loading annotations for {} split from nuScenes version: {}'.format(eval_split, nusc.version))
    # Read out all sample_tokens in DB.
    sample_tokens_all = [s['token'] for s in nusc.sample]
    assert len(sample_tokens_all) > 0, "Error: Database has no samples!"

    # Only keep samples from this split.
    splits = create_splits_scenes()

    # Check compatibility of split with nusc_version.
    version = nusc.version
    if eval_split in {'train', 'val', 'train_detect', 'train_track'}:
        assert version.endswith('trainval'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    elif eval_split in {'mini_train', 'mini_val'}:
        assert version.endswith('mini'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    elif eval_split == 'test':
        assert version.endswith('test'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    else:
        raise ValueError('Error: Requested split {} which this function cannot map to the correct NuScenes version.'
                         .format(eval_split))

    if eval_split == 'test':
        # Check that you aren't trying to cheat :).
        assert len(nusc.sample_annotation) > 0, \
            'Error: You are trying to evaluate on the test set but you do not have the annotations!'

    sample_tokens = []
    for sample_token in sample_tokens_all:
        scene_token = nusc.get('sample', sample_token)['scene_token']
        scene_record = nusc.get('scene', scene_token)
        if scene_record['name'] in splits[eval_split]:
            sample_tokens.append(sample_token)

    dynamic_annotations = EvalBoxes()
    static_annotations = EvalBoxes()
    all_annotations = EvalBoxes()

    # Load annotations and filter predictions and annotations.
    for sample_token in tqdm.tqdm(sample_tokens, leave=verbose):

        sample = nusc.get('sample', sample_token)
        sample_annotation_tokens = sample['anns']

        dynamic_sample_boxes = []
        static_sample_boxes = []
        all_sample_boxes = []

        for sample_annotation_token in sample_annotation_tokens:

            sample_annotation = nusc.get('sample_annotation', sample_annotation_token)

            # Get label name in detection task and filter unused labels.
            detection_name = category_to_detection_name(sample_annotation['category_name'])
            if detection_name is None:
                continue

            # Get attribute_name.
            attr_tokens = sample_annotation['attribute_tokens']
            attr_count = len(attr_tokens)
            if attr_count == 0:
                attribute_name = ''
            elif attr_count == 1:
                attribute_name = attribute_map[attr_tokens[0]]
            else:
                raise Exception('Error: GT annotations must not have more than one attribute!')

            velocity = nusc.box_velocity(sample_annotation['token'])[:2]

            box = box_cls(
                    sample_token=sample_token,
                    translation=sample_annotation['translation'],
                    size=sample_annotation['size'],
                    rotation=sample_annotation['rotation'],
                    velocity=velocity,
                    num_pts=sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts'],
                    detection_name=detection_name,
                    detection_score=-1.0,  # GT samples do not have a score.
                    attribute_name=attribute_name)
            all_sample_boxes.append(box)

            if np.sqrt(velocity[0]**2 + velocity[1]**2) > 0.2:
                dynamic_sample_boxes.append(box)
            else:
                static_sample_boxes.append(box)

        all_annotations.add_boxes(sample_token, all_sample_boxes)
        dynamic_annotations.add_boxes(sample_token, dynamic_sample_boxes)
        static_annotations.add_boxes(sample_token, static_sample_boxes)

    if verbose:
        print("Loaded ground truth annotations for {} samples.".format(len(all_annotations.sample_tokens)))
        print("Loaded dynamic annotations for {} samples.".format(len(dynamic_annotations.sample_tokens)))
        print("Loaded static annotations for {} samples.".format(len(static_annotations.sample_tokens)))

    return all_annotations, dynamic_annotations, static_annotations



def load_dynamic_prediction(result_path: str, max_boxes_per_sample: int, box_cls, verbose: bool = False) \
        -> Tuple[EvalBoxes, EvalBoxes, EvalBoxes, Dict]:
    """
    Loads object predictions from file.
    :param result_path: Path to the .json result file provided by the user.
    :param max_boxes_per_sample: Maximim number of boxes allowed per sample.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The deserialized results and meta data.
    """

    # Load from file and check that the format is correct.
    with open(result_path) as f:
        data = json.load(f)
    assert 'results' in data, 'Error: No field `results` in result file. Please note that the result format changed.' \
                              'See https://www.nuscenes.org/object-detection for more information.'

    # Deserialize results and get meta data.


    # print('Data Results:', type(data['results']), data['results'].keys()) # dict_keys(['sample_tokens', 'boxes'])
    # print(' Data dict', type(data['results']['3e8750f331d7499e9b5123e9eb70f2e2']))
    # print(' Data dict list', type(data['results']['3e8750f331d7499e9b5123e9eb70f2e2'][0]), data['results']['3e8750f331d7499e9b5123e9eb70f2e2'][0].keys())

    dynamic_results = dict()
    static_results = dict()
    for key in data['results'].keys():
        boxes = data['results'][key]
        dynamic_boxes = []
        static_boxes = []
        for box in boxes:
            if np.sqrt(box['velocity'][0]**2 + box['velocity'][1]**2) > 0.2:
                dynamic_boxes.append(box)
            else:
                static_boxes.append(box)
        dynamic_results[key] = dynamic_boxes
        static_results[key] = static_boxes

    all_results = EvalBoxes.deserialize(data['results'], box_cls)
    dynamic_results = EvalBoxes.deserialize(dynamic_results, box_cls)
    static_results = EvalBoxes.deserialize(static_results, box_cls)

    meta = data['meta']
    if verbose:
        print("Loaded results from {}. Found detections for {} samples."
              .format(result_path, len(all_results.sample_tokens)))

    # Check that each sample has no more than x predicted boxes.
    for sample_token in all_results.sample_tokens:
        assert len(all_results.boxes[sample_token]) <= max_boxes_per_sample, \
            "Error: Only <= %d boxes per sample allowed!" % max_boxes_per_sample

    return all_results, dynamic_results, static_results, meta
