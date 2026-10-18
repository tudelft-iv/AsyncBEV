# Representative CMT configurations

This directory contains **six training configurations and six inference configurations**. The focus is CMT + AsyncBEV and CMT + AsyncBEV (FD), covering the paper's three representative delays for each supported modality. All retained models use 0.075 m XY voxels and the VoVNet camera backbone with 1600 × 640 images.

## Training

| Config | Method | Epochs | Learning rate | Samples / GPU |
| --- | --- | ---: | ---: | ---: |
| [fusion/cmt_asyncbev_lidar.py](fusion/cmt_asyncbev_lidar.py) | **CMT + AsyncBEV, delayed LiDAR** | 12 | `2e-5` | 4 |
| [fusion/cmt_asyncbev_lidar_fd.py](fusion/cmt_asyncbev_lidar_fd.py) | **CMT + AsyncBEV (FD), delayed LiDAR** | 12 | `2e-5` | 4 |
| [fusion/cmt_asyncbev_camera.py](fusion/cmt_asyncbev_camera.py) | CMT + AsyncBEV, delayed camera | 12 | `2e-5` | 4 |
| [fusion/cmt.py](fusion/cmt.py) | Synchronous CMT | 20 | `1e-4` | 4 |
| [fusion/cmt_da_lidar.py](fusion/cmt_da_lidar.py) | Asynchronous data augmentation (DA) | 12 | `2e-5` | 4 |
| [fusion/cmt_sensor_staleness_lidar.py](fusion/cmt_sensor_staleness_lidar.py) | Sensor-staleness feature baseline | 12 | `2e-4` | 4 |

The main AsyncBEV configs freeze the backbones and train the alignment module and detection head. FD means **frozen detector**: `freeze_backbone=False`, `freeze_detector=True`. Its historical training filename used `FT`; the new name follows the paper's terminology. The camera config preserves the 12-epoch, `2e-5` setting associated with the main camera checkpoint, rather than a different 18-epoch schedule.

These are representative method configurations, not byte-for-byte copies of historical resolved runs. In particular, the retained sensor-staleness schedule is not independently established as the schedule that produced its paper checkpoint. Batch size and distributed world size must also match when reproducing training.

Experiment IDs and output directories use the config names. `resume_from=None` avoids restoring an old experiment implicitly. Training initialization remains in `load_from`; set the intended checkpoint and use the [training commands](../../../README.md#training). Logging hooks include W&B and should be configured for your account or removed if unused.

The synchronous `fusion/cmt.py` baseline uses `checkpoints/CMT/fcos3d_vovnet_imgbackbone-remapped.pth` and enables joint image/point GT database sampling. Prepare its `nuscenes_dbinfos_train.pkl` and referenced point/image crops as described in [data preparation](../../../README.md#data-preparation). The AsyncBEV/FD configs disable that sampling stage and initialize from `checkpoints/CMT/voxel0075_vov_1600x640_epoch20.pth`.

## Inference and evaluation

| Config | Delayed modality | Paper delay label | Effective `async_frames` |
| --- | --- | --- | ---: |
| [inference/cmt_asyncbev_lidar_0.py](inference/cmt_asyncbev_lidar_0.py) | LiDAR | 0 s | `0` |
| [inference/cmt_asyncbev_lidar_5.py](inference/cmt_asyncbev_lidar_5.py) | LiDAR | 0.25 s | `5` |
| [inference/cmt_asyncbev_lidar_10.py](inference/cmt_asyncbev_lidar_10.py) | LiDAR | 0.5 s | `-1` |
| [inference/cmt_asyncbev_camera_0.py](inference/cmt_asyncbev_camera_0.py) | Camera | 0 s | `0` |
| [inference/cmt_asyncbev_camera_3.py](inference/cmt_asyncbev_camera_3.py) | Camera | 0.25 s | `3` |
| [inference/cmt_asyncbev_camera_7.py](inference/cmt_asyncbev_camera_7.py) | Camera | 0.5 s | `-1` |

**The three LiDAR inference configs serve both the main AsyncBEV and FD checkpoints.** Their model architectures match; FD changes training freezing flags. Select the matching trained checkpoint explicitly as the second argument to `tools/test.py`. Inference configs have `load_from=None` and `resume_from=None`; output directories default to `outputs/eval/`.

The suffixes are protocol labels. `-1` selects the last available history entry; actual offsets are determined by sensor timestamps. Changing only top-level `eval_async_frame` through `--cfg-options` does not reconstruct the nested pipeline. Use the corresponding supplied config, or update all effective loader settings.

These configs run prediction on the nuScenes validation split through `test_pipeline` / `data.test`. They still contain `LoadGTVoxelFlow`, and the current forward path accesses flow metadata. **The prepared flow supplement is required even for inference; these are not label-free deployment configs.** The training files also contain test pipelines for their default validation setting.

For example, from the repository root:

```bash
python tools/test.py projects/CMT/configs/inference/cmt_asyncbev_lidar_5.py \
  /absolute/path/to/asyncbev_lidar_or_fd_checkpoint.pth \
  --gpu-id 0 --eval-dynamic --work-dir outputs/eval/asyncbev_lidar_025
```

Use `--eval bbox` for ordinary all-object metrics, or `--out /path/to/predictions.pkl` for prediction output without metric evaluation. Follow your execution environment's GPU and job-management rules. See the [evaluation guide](../../../README.md#evaluation) for output artifacts and distributed commands.

## Checkpoint correspondence

The strongest existing evaluation-log associations are:

| Method | Confirmed checkpoint filename | Paper correspondence | Evidence limit |
| --- | --- | --- | --- |
| **CMT + AsyncBEV, LiDAR** | `cmt_asyncbev_lidar_epoch12.pth` | Table 1; delayed-LiDAR row in Table IV | Main absolute NDS/mAP cells match the audited saved results. |
| **CMT + AsyncBEV (FD), LiDAR** | `cmt_asyncbev_lidar_fd_epoch10.pth` | Table 1, FD row | Best-supported evaluation family; not an exact match for every printed cell. |
| CMT + AsyncBEV, camera | `cmt_asyncbev_camera_epoch12.pth` | Table III | Main absolute NDS/mAP cells match the audited saved results. |

These filenames distinguish three separate training runs while preserving the original checkpoint bytes. See the [checkpoint guide](../../../README.md#checkpoints) for public SURFdrive download links, placement, and SHA-256 verification. The three file links require no login or password; the checkpoints are not published on Hugging Face. At 0.5 s delay, saved all-object NDS/mAP (%) are **69.99 / 66.40** (LiDAR), **68.70 / 64.40** (FD), and **72.28 / 69.05** (camera). These are historical saved results, not new evaluations of the cleaned configs.

For FD, saved 0 s All NDS is `73.07392841%` (direct rounding: `73.1`, paper: `73.0`); saved 0 s Static NDS is `69.25514126%` (direct rounding: `69.3`, paper: `69.2`). At 0.25 s, Static mAP is `59.84573653%` (direct rounding: `59.8`, paper: `59.9`, explainable by two-stage rounding). Keep these discrepancies visible rather than selecting a different epoch to match one cell.

## Scope of the compact set

The previous 97-file collection has been reduced to these 12 entrypoints. Machine-specific copies, demos, alternative camera schedules, unimodal/model-size variants, dense intermediate-delay sweeps, noise experiments, and ablations are omitted. The retained inference configs cover the main AsyncBEV/FD settings; they do not provide the full CMT/EMC/DA/sensor-staleness baseline delay matrix or all ablation curves. The corresponding model implementations remain in the repository.
