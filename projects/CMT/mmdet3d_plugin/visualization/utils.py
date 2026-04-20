from typing import List, Dict, Any

import numpy as np
from pyquaternion import Quaternion

from matplotlib import pyplot as plt
from nuscenes import NuScenes

from nuscenes.eval.common.data_classes import EvalBox, EvalBoxes

from nuscenes.eval.common.utils import center_distance ## use self-defined boxes-to-sensor
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from .vis_classes import boxes_to_sensor

def visualize_sample_custom(nusc: NuScenes,
                            sample_token: str,
                            gt_boxes: EvalBoxes,
                            pred_boxes: EvalBoxes,
                            nsweeps: int = 3,
                            conf_th: float = 0.15 ,
                            class_names=None,
                            eval_range: float = 30,
                            show_GT = True,
                            show_mode = 'all',
                            verbose: bool = True,
                            savepath: str = None) -> None:
    """
        Visualizes a sample from BEV with annotations and detection results.
        :param nusc: NuScenes object.
        :param sample_token: The nuScenes sample token.
        :param gt_boxes: Ground truth boxes grouped by sample.
        :param pred_boxes: Prediction grouped by sample.
        :param nsweeps: Number of sweeps used for lidar visualization.
        :param conf_th: The confidence threshold used to filter negatives.
        :param eval_range: Range in meters beyond which boxes are ignored.
        :param verbose: Whether to print to stdout.
        :param savepath: If given, saves the the rendering here instead of displaying.
        """
    # Retrieve sensor & pose records.
    sample_rec = nusc.get('sample', sample_token)
    sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    # Get boxes.
    boxes_gt_global = gt_boxes[sample_token]
    boxes_est_global = pred_boxes[sample_token]

    if conf_th == 'auto':
        predict_boxes = get_boxes_auto_conf_th(boxes_gt_global, boxes_est_global, class_names, center_distance, dist_th=2)
    else:
        assert conf_th is not None and isinstance(conf_th, float), f'Error: Confidence threshold must be a float, but get {type(conf_th)}!'
        predict_boxes = get_boxes_conf_th(boxes_gt_global, boxes_est_global, class_names, center_distance, conf_th=conf_th, dist_th=2)

    tp_predict_boxes = []
    fp_predict_boxes = []
    for pred_boxes_cls in predict_boxes:
        tp_predict_boxes.extend(pred_boxes_cls['tp_boxes'])
        fp_predict_boxes.extend(pred_boxes_cls['fp_boxes'])

    # Map GT boxes to lidar.
    boxes_gt = boxes_to_sensor(boxes_gt_global, pose_record, cs_record)

    # Map TP and FP EST boxes to lidar.
    tp_boxes_est = boxes_to_sensor(tp_predict_boxes, pose_record, cs_record)
    fp_boxes_est = boxes_to_sensor(fp_predict_boxes, pose_record, cs_record)

    # Get point cloud in lidar frame.
    pc, _ = LidarPointCloud.from_file_multisweep(nusc, sample_rec, 'LIDAR_TOP', 'LIDAR_TOP', nsweeps=nsweeps)

    # Init axes.
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), tight_layout=True, frameon=True)
    ax.set_facecolor('white')
    ax.tick_params(left = False,
                   right = False ,
                   labelleft = False ,
                   labelbottom = False,
                   bottom = False)
    # Show point cloud.
    points = view_points(pc.points[:3, :], np.eye(4), normalize=False)
    dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
    colors = np.minimum(1, dists / eval_range)
    ax.scatter(points[0, :], points[1, :], c='dimgray', s=0.2)

    # Show ego vehicle.
    ax.plot(0, 0, 'x', color='black')

    GT_BOX_LINEWIDTH = 4
    TP_BOX_LINEWIDTH = 2
    FP_BOX_LINEWIDTH = 2

    DYNAMIC_LINESTYLE = '-' # dotted line
    STATIC_LINESTYLE = '-' # solid line

    GT_BOX_COLOR = 'dodgerblue'
    TP_BOX_COLOR = 'limegreen'
    FP_BOX_COLOR = 'red'
    # Show GT boxes.
    assert show_mode in ['all', 'dynamic', 'static'], f'Error: Invalid show mode {show_mode}!'
    if show_GT:
        for box in boxes_gt:
            if show_mode =='all':
                if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
                    box.render(ax, view=np.eye(4), colors=(GT_BOX_COLOR, GT_BOX_COLOR, GT_BOX_COLOR), linewidth=GT_BOX_LINEWIDTH, linestyle=DYNAMIC_LINESTYLE)
                else:
                    box.render(ax, view=np.eye(4), colors=(GT_BOX_COLOR, GT_BOX_COLOR, GT_BOX_COLOR), linewidth=GT_BOX_LINEWIDTH, linestyle=STATIC_LINESTYLE)
            elif show_mode == 'dynamic':
                if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
                    box.render(ax, view=np.eye(4), colors=(GT_BOX_COLOR, GT_BOX_COLOR, GT_BOX_COLOR), linewidth=GT_BOX_LINEWIDTH, linestyle=DYNAMIC_LINESTYLE)
            elif show_mode == 'static':
                if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) <= 0.2:
                    box.render(ax, view=np.eye(4), colors=(GT_BOX_COLOR, GT_BOX_COLOR, GT_BOX_COLOR), linewidth=GT_BOX_LINEWIDTH, linestyle=STATIC_LINESTYLE)

    for tp_box in tp_boxes_est:
        if show_mode == 'all':
            if np.sqrt(tp_box.velocity[0] ** 2 + tp_box.velocity[1] ** 2) > 0.2:
                tp_box.render(ax, view=np.eye(4), colors=(TP_BOX_COLOR, TP_BOX_COLOR, TP_BOX_COLOR), linewidth=TP_BOX_LINEWIDTH, linestyle=DYNAMIC_LINESTYLE)
            else:
                tp_box.render(ax, view=np.eye(4), colors=(TP_BOX_COLOR, TP_BOX_COLOR, TP_BOX_COLOR), linewidth=TP_BOX_LINEWIDTH, linestyle=STATIC_LINESTYLE)
        elif show_mode == 'dynamic':
            if np.sqrt(tp_box.velocity[0] ** 2 + tp_box.velocity[1] ** 2) > 0.2:
                tp_box.render(ax, view=np.eye(4), colors=(TP_BOX_COLOR, TP_BOX_COLOR, TP_BOX_COLOR), linewidth=TP_BOX_LINEWIDTH, linestyle=DYNAMIC_LINESTYLE)
        elif show_mode == 'static':
            if np.sqrt(tp_box.velocity[0] ** 2 + tp_box.velocity[1] ** 2) <= 0.2:
                tp_box.render(ax, view=np.eye(4), colors=(TP_BOX_COLOR, TP_BOX_COLOR, TP_BOX_COLOR), linewidth=TP_BOX_LINEWIDTH, linestyle=STATIC_LINESTYLE)

    for fp_box in fp_boxes_est:
        if show_mode == 'all':
            if np.sqrt(fp_box.velocity[0] ** 2 + fp_box.velocity[1] ** 2) > 0.2:
                fp_box.render(ax, view=np.eye(4), colors=(FP_BOX_COLOR, FP_BOX_COLOR, FP_BOX_COLOR), linewidth=FP_BOX_LINEWIDTH, linestyle=DYNAMIC_LINESTYLE)
            else:
                fp_box.render(ax, view=np.eye(4), colors=(FP_BOX_COLOR, FP_BOX_COLOR, FP_BOX_COLOR), linewidth=FP_BOX_LINEWIDTH, linestyle=STATIC_LINESTYLE)
        elif show_mode == 'dynamic':
            if np.sqrt(fp_box.velocity[0] ** 2 + fp_box.velocity[1] ** 2) > 0.2:
                fp_box.render(ax, view=np.eye(4), colors=(FP_BOX_COLOR, FP_BOX_COLOR, FP_BOX_COLOR), linewidth=FP_BOX_LINEWIDTH, linestyle=DYNAMIC_LINESTYLE)
        elif show_mode == 'static':
            if np.sqrt(fp_box.velocity[0] ** 2 + fp_box.velocity[1] ** 2) <= 0.2:
                fp_box.render(ax, view=np.eye(4), colors=(FP_BOX_COLOR, FP_BOX_COLOR, FP_BOX_COLOR), linewidth=FP_BOX_LINEWIDTH, linestyle=STATIC_LINESTYLE)
        # fp_box.render(ax, view=np.eye(4), colors=('red', 'red', 'red'), linewidth=1)
    # Show EST boxes.
    # for box in boxes_est:
    #     # Show only predictions with a high score.
    #     assert not np.isnan(box.score), 'Error: Box score cannot be NaN!'
    #     if box.score >= conf_th:
    #         box.render(ax, view=np.eye(4), colors=('red', 'red', 'red'), linewidth=1)

    # Limit visible range.
    axes_limit = 23 + 4  # Slightly bigger to include boxes that extend beyond the range.
    # axes_limit = eval_range
    major_ticks = np.arange(-54, 54.1, 13.5)
    minor_ticks = np.arange(-54, 54.1, 2.7)
    # ax.set_xticks(major_ticks, )
    # ax.set_xticks(minor_ticks, minor=True)
    # ax.set_yticks(major_ticks)
    # ax.set_yticks(minor_ticks, minor=True)
    # ax.grid(which='minor', alpha=0.5)
    # ax.grid(which='major', alpha=0.6, linewidth=1.5)
    ax.set_xlim(-axes_limit, axes_limit)
    ax.set_ylim(-axes_limit, axes_limit)

    # Show / save plot.
    if verbose:
        print('Rendering sample token %s' % sample_token)
    # plt.title(sample_token)
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0)
        #plt.show()
        plt.close()
    else:
        plt.show()
    return fig

def get_boxes_auto_conf_th(gt_boxes, pred_boxes, class_names, dist_fcn_callable, dist_th=2):
    conf_th_list = [0.15, 0.2, 0.3, 0.4, 0.5]
    filter_boxes_list = []
    for class_name in class_names:
        f1_score_list_per_class = []
        boxes_list_per_class = []
        for conf_th in conf_th_list:
            npos = len([1 for gt_box in gt_boxes if gt_box.detection_name == class_name])
            pred_boxes_list = [box for box in pred_boxes if box.detection_name == class_name]
            pred_boxes_list = [box for box in pred_boxes_list if box.detection_score > conf_th]
            pred_confs = [box.detection_score for box in pred_boxes_list]
            sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

            tp = []  # Accumulator of true positives.
            fp = []  # Accumulator of false positives.
            conf = []  # Accumulator of confidences.

            taken = set()
            tp_boxes = []
            fp_boxes = []
            for ind in sortind:
                pred_box = pred_boxes_list[ind]
                min_dist = np.inf
                match_gt_idx = None

                for gt_idx, gt_box in enumerate(gt_boxes):
                    if gt_box.detection_name == class_name and not gt_idx in taken:
                        this_distance = dist_fcn_callable(gt_box, pred_box)
                        if this_distance < min_dist:
                            min_dist = this_distance
                            match_gt_idx = gt_idx

                is_match = min_dist < dist_th

                if is_match:
                    taken.add(match_gt_idx)
                    tp_boxes.append(pred_box)
                    tp.append(1)
                    fp.append(0)
                else:
                    tp.append(0)
                    fp.append(1)
                    fp_boxes.append(pred_box)

            tp = np.sum(tp).astype(float)
            fp = np.sum(fp).astype(float)

            precision = tp / (fp + tp)
            recall = tp / float(npos)

            F1 = 2*(precision * recall)/(precision + recall)
            f1_score_list_per_class.append(F1)
            boxes_list_per_class.append((tp_boxes, fp_boxes))

        filter_boxes_list.append(dict(class_name= class_name,
                                      tp_boxes = boxes_list_per_class[np.argmax(f1_score_list_per_class)][0],
                                      fp_boxes = boxes_list_per_class[np.argmax(f1_score_list_per_class)][1]))

    return filter_boxes_list


def get_boxes_conf_th(gt_boxes, pred_boxes, class_names, dist_fcn_callable, conf_th=0.15, dist_th=2):
    filter_boxes_list = []
    for class_name in class_names:
        f1_score_list_per_class = []
        boxes_list_per_class = []
        npos = len([1 for gt_box in gt_boxes if gt_box.detection_name == class_name])
        pred_boxes_list = [box for box in pred_boxes if box.detection_name == class_name]
        pred_boxes_list = [box for box in pred_boxes_list if box.detection_score > conf_th]
        pred_confs = [box.detection_score for box in pred_boxes_list]
        sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

        tp = []  # Accumulator of true positives.
        fp = []  # Accumulator of false positives.

        taken = set()
        tp_boxes = []
        fp_boxes = []
        for ind in sortind:
            pred_box = pred_boxes_list[ind]
            min_dist = np.inf
            match_gt_idx = None

            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_box.detection_name == class_name and not gt_idx in taken:
                    this_distance = dist_fcn_callable(gt_box, pred_box)
                    if this_distance < min_dist:
                        min_dist = this_distance
                        match_gt_idx = gt_idx

            is_match = min_dist < dist_th

            if is_match:
                taken.add(match_gt_idx)
                tp_boxes.append(pred_box)
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)
                fp_boxes.append(pred_box)

        filter_boxes_list.append(dict(class_name= class_name,
                                      tp_boxes = tp_boxes,
                                      fp_boxes = fp_boxes))

    return filter_boxes_list

def visualize_sample_custom_naive(nusc: NuScenes,
                            sample_token: str,
                            gt_boxes: EvalBoxes,
                            pred_boxes: EvalBoxes,
                            nsweeps: int = 3,
                            conf_th: float = 0.15,
                            eval_range: float = 50,
                            show_GT = True,
                            verbose: bool = True,
                            savepath: str = None) -> None:
    """
        Visualizes a sample from BEV with annotations and detection results.
        :param nusc: NuScenes object.
        :param sample_token: The nuScenes sample token.
        :param gt_boxes: Ground truth boxes grouped by sample.
        :param pred_boxes: Prediction grouped by sample.
        :param nsweeps: Number of sweeps used for lidar visualization.
        :param conf_th: The confidence threshold used to filter negatives.
        :param eval_range: Range in meters beyond which boxes are ignored.
        :param verbose: Whether to print to stdout.
        :param savepath: If given, saves the the rendering here instead of displaying.
        """
    # Retrieve sensor & pose records.
    sample_rec = nusc.get('sample', sample_token)
    sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    # Get boxes.
    boxes_gt_global = gt_boxes[sample_token]
    boxes_est_global = pred_boxes[sample_token]

    # Map GT boxes to lidar.
    boxes_gt = boxes_to_sensor(boxes_gt_global, pose_record, cs_record)

    # Map EST boxes to lidar.
    boxes_est = boxes_to_sensor(boxes_est_global, pose_record, cs_record)

    # Add scores to EST boxes.
    for box_est, box_est_global in zip(boxes_est, boxes_est_global):
        box_est.score = box_est_global.detection_score

    # Get point cloud in lidar frame.
    pc, _ = LidarPointCloud.from_file_multisweep(nusc, sample_rec, 'LIDAR_TOP', 'LIDAR_TOP', nsweeps=nsweeps)

    # Init axes.
    _, ax = plt.subplots(1, 1, figsize=(9, 9))
    ax.set_facecolor('black')
    ax.tick_params(left = False,
                   right = False ,
                   labelleft = False ,
                   labelbottom = False,
                   bottom = False)
    # Show point cloud.
    points = view_points(pc.points[:3, :], np.eye(4), normalize=False)
    dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
    # colors = np.minimum(1, dists / eval_range)
    ax.scatter(points[0, :], points[1, :], c='gainsboro', s=0.2)

    # Show ego vehicle.
    ax.plot(0, 0, 'x', color='black')

    # Show GT boxes.
    if show_GT:
        for box in boxes_gt:
            box.render(ax, view=np.eye(4), colors=('lime', 'lime', 'lime'), linewidth=2)

    # Show EST boxes.
    for box in boxes_est:
        # Show only predictions with a high score.
        assert not np.isnan(box.score), 'Error: Box score cannot be NaN!'
        if box.score >= conf_th:
            box.render(ax, view=np.eye(4), colors=('red', 'red', 'red'), linewidth=1)

    # Limit visible range.
    axes_limit = eval_range + 3  # Slightly bigger to include boxes that extend beyond the range.
    ax.set_xlim(-axes_limit, axes_limit)
    ax.set_ylim(-axes_limit, axes_limit)

    # Show / save plot.
    if verbose:
        print('Rendering sample token %s' % sample_token)
    # plt.title(sample_token)
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0)
        #plt.show()
        plt.close()
    else:
        plt.show()





