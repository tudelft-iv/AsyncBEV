# Training, evaluation, and inspection tools

These tools accompany the CMT implementation in `projects/CMT`. Use the
[root README](../README.md) for the environment, data, checkpoint requirements,
and complete training/evaluation examples. Run those commands from the
repository root with the documented MMCV 1.x / MMDetection3D 1.0.0rc5 environment
activated. The standalone flow-data tools have separate CPU dependencies,
documented in their [guide](data_converter/README.md).

## Entry points

| File | Purpose |
| --- | --- |
| [train.py](train.py) | Training through the installed `mmdet3d.apis.train_model`, with plugin registration and explicit resume controls. |
| [test.py](test.py) | One inference pipeline for ordinary `--eval bbox`, `--eval-dynamic`, raw prediction export, or `--format-only`. |
| [dist_train.sh](dist_train.sh) | `bash tools/dist_train.sh CONFIG GPUS [options...]`. |
| [dist_test.sh](dist_test.sh) | `bash tools/dist_test.sh CONFIG CHECKPOINT GPUS [options...]`; accepts `--eval-dynamic`. |
| [misc/print_config.py](misc/print_config.py) | Display the fully resolved config and optional overrides without constructing a model. |
| [analysis_tools/analyze_logs.py](analysis_tools/analyze_logs.py) | Summarize training timing or plot exact metric keys from MMCV JSON-lines logs. |
| [data_converter/generate_async_flow.py](data_converter/generate_async_flow.py) | CPU camera/LiDAR grid-flow generation; see the [data-tool guide](data_converter/README.md) for metadata/maps and output conventions. |
| [data_converter/repair_pts_in_boxes.py](data_converter/repair_pts_in_boxes.py) | Read-only legacy-label checking by default; `--apply` writes validated copies to a separate tree. |

Training resume choices are mutually exclusive: `--no-resume` starts with the
configured `load_from` weights, `--resume-from FILE` restores a checkpoint's
training state, and `--auto-resume` searches the work directory. Without one of
these flags, the config's resume settings remain active. Select a fresh work
directory for a new experiment and inspect logging hooks before starting.

`test.py --eval-dynamic` implies `--eval bbox` and uses the existing
`CustomNuScenesDataset.evaluate_dynamic` protocol. It does not change the
static/dynamic threshold or the underlying metric calculations. The default
evaluation directory is `outputs/eval/<config stem>/<checkpoint stem>`;
`--work-dir` overrides it. The entrypoint writes:

- `resolved_config.py`: config after overrides and test-pipeline preparation.
- `run_info.json`: CLI arguments, checkpoint, evaluator options, seed, and
  process/GPU information.
- `metrics.json`: returned metrics, for an evaluation operation.
- `results/`: formatted nuScenes predictions and evaluator files; for the
  usual `pts_bbox` head, dynamic summaries are under `results/pts_bbox/`.

Use distinct output directories across runs. `--eval-options
jsonfile_prefix=...` overrides the results directory. `--out FILE.pkl` also
saves raw model outputs; it is optional. `--format-only` produces submission
JSON without metrics. Visualization options require non-distributed inference.

Launchers preserve the caller's GPU visibility, CPU-thread settings, and NCCL
settings. They support `PYTHON` (an executable path), `PORT` (default 29500),
`NNODES` (default 1), `NODE_RANK` (default 0), and `MASTER_ADDR` (default
127.0.0.1). Multi-node jobs need the same code/data and reachable master
address on every node. `GPUS` means processes per node, one per visible GPU.
These wrappers use `torch.distributed.launch` for the documented PyTorch
1.9/1.11 versions; the Python entrypoints accept both local-rank spellings.

## Configuration and log inspection

```bash
CONFIG=projects/CMT/configs/fusion/cmt_asyncbev_lidar.py
python tools/misc/print_config.py "$CONFIG" \
  --cfg-options data.workers_per_gpu=2 --out /tmp/asyncbev-resolved.py

# Substitute the actual JSON-lines log produced by your training run.
LOG=outputs/train/asyncbev_lidar/RUN_TIMESTAMP.log.json
python tools/analysis_tools/analyze_logs.py cal_train_time "$LOG"

# Optional plotting dependency, unnecessary for timing analysis.
python -m pip install matplotlib
python tools/analysis_tools/analyze_logs.py plot_curve "$LOG" \
  --keys loss --backend Agg --out loss.png
python tools/analysis_tools/analyze_logs.py plot_curve "$LOG" \
  --mode eval --keys pts_bbox_NuScenes/NDS --backend Agg --out nds.png
```

Use metric keys actually present in the log. Validation curves use epoch
numbers; training curves use logged-record indices so partial or resumed
epochs do not invent global iteration counts. Timing excludes the first
logged sample in each epoch by default; `--include-outliers` includes it.
Timing analysis needs only Python's standard library. Plotting needs
Matplotlib; the optional `--style` argument also needs seaborn.

Config inspection needs MMCV but does not import the detector/plugin. Overrides
apply after the Python config has been evaluated, so changing a top-level
`data_root` or `eval_async_frame` does not reconstruct nested loader arguments.

## Source selection and changes

The entrypoints were adapted from [CMT](https://github.com/junjie18/CMT) /
[MMDetection3D](https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0rc5)
tools and the AsyncBEV development variants `tools/cmt/train_cmt.py`,
`test_cmt.py`, and `test_cmt_dynamic.py`. Existing OpenMMLab copyright headers
are retained. The ordinary and dynamic test scripts are combined to avoid
maintaining duplicate inference code. The config inspector and log analyzer
come from the corresponding OpenMMLab tool paths.

The migration fixes plugin-directory handling, an unnecessary nested-package
import, non-distributed rank access, distributed learning-rate scaling order,
explicit resume overrides, model device placement, persistent evaluation
outputs, and shell argument quoting/resource overrides. The dataset wrapper
also preserves all result heads in each all/static/dynamic subset. Default
single-head metric definitions are unchanged. Log analysis now handles
partial/resumed epochs and mixed training/validation records.

Development-only cluster jobs, duplicate tools, notebooks, and visualization
experiments were omitted. Portable [flow tools](data_converter/README.md)
consolidate the historical LiDAR/camera generators and point-field postprocessor.
They reuse [standalone CMT geometry](../projects/CMT/flow_geometry.py), retain the
original helper import API, and remove legacy UniBEV imports and host paths.
Generation includes `pts_in_boxes`; repair preserves saved flow and checks its
agreement with reconstructed points before producing separate copies. The
custom input-map builders and full-release semantic validation remain separate
work. Use the published Hugging Face supplement for existing experiments.

## Checks

```bash
python -m unittest discover -s tests -v
bash -n tools/dist_train.sh tools/dist_test.sh
git diff --check
```

Tests exercise CLI routing, resume policy, plugin imports, distributed
argument/environment forwarding, result-head aggregation, synthetic log
analysis, point-loader timestamp/coordinate behavior, and synthetic flow-tool
geometry, map, CLI, and file-preservation behavior (requiring NumPy).
ML dependencies are stubbed where needed. They do not establish a
working CUDA installation, successful training/inference, or reproduction of
the paper's metrics. Full dataset/model execution remains a separate check.
