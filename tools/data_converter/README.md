# GT flow generation and point-field repair

These CPU tools generate camera/LiDAR grid-flow labels and recover the missing
`pts_in_boxes` field in legacy labels. They reuse the CMT geometry in
[projects/CMT/flow_geometry.py](../../projects/CMT/flow_geometry.py), without
importing the detector, Torch, MMCV, or FlashAttention. The former plugin helper
path re-exports the same geometry functions for existing callers.

| Entry point | Purpose |
| --- | --- |
| [generate_async_flow.py](generate_async_flow.py) | Generate camera or LiDAR labels, including `pts_in_boxes`, into an explicit output directory. |
| [repair_pts_in_boxes.py](repair_pts_in_boxes.py) | Check existing labels and optionally write validated, repaired copies into a separate directory. |
| [async_flow.py](async_flow.py) | Shared map resolution, schema checks, geometry comparison, and atomic file creation. |

## Inputs and dependencies

Use Python 3.8 or newer with **NumPy 1.x, nuScenes-devkit, pyquaternion**, and the
SDK's dependencies installed. These normally accompany the prepared CMT
environment. For a separate CPU environment, install them with
`python -m pip install "numpy<2" nuscenes-devkit pyquaternion`; do not upgrade a
working CMT environment just to run these tools. `--help` requires only Python's
standard library. Use the same SDK version and metadata as the original labels
when repairing them.

Use NumPy 1.x when producing labels for the legacy CMT environment: NumPy 2.x
pickles can reference modules unavailable to its NumPy 1.x readers. The tools
reject new label serialization under NumPy 2.x; read-only checks and unchanged
byte copies remain available. New label
files use pickle protocol 4, matching the historical Python 3.8 writer default;
already-complete repair copies retain their original bytes.

Required inputs are:

- The nuScenes root, including the matching `v1.0-trainval/` or `v1.0-mini/`
  JSON metadata and SDK-required map assets. The grid-flow calculations do not
  read camera images or measured LiDAR point clouds.
- A trusted AsyncBEV camera or LiDAR map PKL. The two custom maps are available
  in the [published supplement](https://huggingface.co/datasets/ShimingWang/AsyncBEV/tree/b31cfeec498070875c4dfdd531e6aa167d5b229e).
- For repair, the existing label directory containing
  `<sample_token>/<async_data_token>.pkl`.

These scripts consume the custom maps; they **do not create those maps** from
raw nuScenes. Standard MMDetection3D info-file sweep generation is a separate
step. Only load trusted pickle files.

Map keys are matched to metadata by the full nuScenes-relative path, such as
`samples/CAM_BACK_LEFT/filename.jpg`. Historical absolute prefixes and
`data/nuscenes/` prefixes therefore need no source edits. The tool does not
guess by basename, rewrite the map, or open its stored sensor paths. Unknown
paths, duplicate output pairs, wrong sensor/scene/timestamp references, and
unknown selected sample tokens fail validation. Use a map matching the selected
metadata version; `--limit` does not hide an incompatible map.

## Generate labels

Run from the repository root, replacing the paths below. Outputs are label
roots, so a generated file is stored directly under
`OUTPUT/<sample_token>/<async_data_token>.pkl`.

```bash
DATA_ROOT=/path/to/nuscenes
CAM_MAP="$DATA_ROOT/async_supplement/cams_sample_sweeps_map_ego_sample_w_ego_pose_timestamp.pkl"
LIDAR_MAP="$DATA_ROOT/async_supplement/lidar_sample_sweeps_map_w_extrin_ego_sample_ref_coord.pkl"

# Compute a small selection without writing labels.
python tools/data_converter/generate_async_flow.py \
  --data-root "$DATA_ROOT" --map-file "$CAM_MAP" --modality camera \
  --output-dir outputs/flow-generated/async_cam_flow --limit 10 --dry-run

# Generate camera labels into a new tree.
python tools/data_converter/generate_async_flow.py \
  --data-root "$DATA_ROOT" --map-file "$CAM_MAP" --modality camera \
  --output-dir outputs/flow-generated/async_cam_flow --workers 4 \
  --report outputs/flow-generated/camera-run.json

# Generate LiDAR labels into a separate new tree.
python tools/data_converter/generate_async_flow.py \
  --data-root "$DATA_ROOT" --map-file "$LIDAR_MAP" --modality lidar \
  --output-dir outputs/flow-generated/async_lidar_flow --workers 4 \
  --report outputs/flow-generated/lidar-run.json
```

Existing destination files cause errors by default. `--existing validate`
supports resuming a partially completed tree: it checks tokens, required
fields, shapes, finite values, reconstructed points, and both recomputed flows
before retaining an existing file. It never overwrites it. This also means
resume validation performs geometry work. A malformed old camera label should
be handled by the repair command, not silently skipped by generation.

## Check and repair missing pts_in_boxes

The repair command is **read-only by default**. It reconstructs points using
the selected metadata/grid and recomputes scene/pose flow for comparison with
the existing arrays (`rtol=1e-5`, `atol=1e-6`). Matching shape alone is not
sufficient. A disagreement is reported as an error, without writing that label.

```bash
# Check a small selection; this does not modify the input labels.
python tools/data_converter/repair_pts_in_boxes.py \
  --data-root "$DATA_ROOT" --map-file "$CAM_MAP" \
  --input-dir "$DATA_ROOT/async_supplement/async_cam_flow" --limit 10

# When ready, write repaired copies into a separate output tree.
python tools/data_converter/repair_pts_in_boxes.py \
  --data-root "$DATA_ROOT" --map-file "$CAM_MAP" \
  --input-dir "$DATA_ROOT/async_supplement/async_cam_flow" \
  --output-dir outputs/flow-repaired/async_cam_flow --apply --workers 4 \
  --report outputs/flow-repaired/camera-run.json
```

Camera is the repair default; use `--modality lidar --map-file "$LIDAR_MAP"`
for LiDAR labels. A missing source file is an error; repair does not fabricate
new flow. Existing scene/pose arrays, their dtypes, tokens, and extra metadata
are retained. Only a missing `pts_in_boxes` key is added. Already-complete
validated labels are copied byte-for-byte, so the output contains the entire
selected set. An incomplete selection is not a full replacement dataset.

Input and output directories must be separate and must not contain one another;
there is no in-place option. Output collisions use the same `--existing` policy
as generation; repair resume additionally compares all fields with the expected
source copy. Label files are published atomically with a same-filesystem hard
link that refuses an existing destination. Filesystems must support this
operation. A failed file does not roll back other successfully created files.

## Shared options and scientific conventions

- `--version`: `v1.0-trainval` by default, or `v1.0-mini`; GT annotation-free
  nuScenes test data is not supported.
- `--sample-token TOKEN`: repeat to restrict source samples. `--limit N` limits
  selected label pairs after validation, sorted by sample/async token.
- `--workers N`: CPU processes, default **1**. Each process loads its own SDK
  metadata and grid; choose a worker count that fits memory. Caller BLAS thread
  settings are preserved; unset thread counts default to one.
- `--bev-size 200 200 --resolution 0.5
  --point-cloud-range -54 -54 -5 54 54 3`: original defaults. The historical
  normalized grid has `H=int(h/resolution)`, `W=int(w/resolution)`, and
  `Z=int(z_extent/resolution)`, followed by scaling into the point-cloud range.
  **Resolution 0.5 does not mean physical 0.5 m XY spacing.** Keep the original
  endpoints, flattening order, and parameters when reconstructing saved rows.
- `--report FILE`: create a new JSON report with effective parameters, dependency
  versions, input-map and geometry-code SHA-256, counts, elapsed time, and up to
  100 per-file errors. Reports never
  overwrite existing files; use a new filename on each run. Dry-run does not
  write labels but can write this explicitly requested report.

Camera generation uses **CAM_BACK_LEFT** timestamps, matching `LoadGTVoxelFlow`.
Coordinates are synthetic grid points selected inside synchronous LiDAR GT
boxes, not image depth. Box order and duplicated points in overlapping boxes
are preserved. No boxes produces the historical single zero row; boxes with no
selected grid points can produce an empty array. Scene displacement and pose
displacement are stored separately; camera targets retain source LiDAR
extrinsics at the target ego pose.

Camera labels use `pose_flow_s2t` plus the historical camera token keys;
LiDAR labels use `pose_flow_sync2async` and LiDAR token keys. Both include
`pts_in_boxes` and `gt_flow_sync2async`. Repair accepts either pose-flow alias,
but rejects disagreement when both are present. Additional metadata is allowed.
Agreement with recomputed flow is a consistency check, not proof of historical
execution provenance or an alternative to matching the original grid/metadata.

Per-file failures are reported while other selected pairs continue; the final
exit code is nonzero if any file fails. Successful completion validates only
the selected pairs. Paths supplied on the CLI are relative to the caller's
working directory; script paths can be absolute when invoked elsewhere.

## Validation and provenance

The tools consolidate the historical multiprocessing LiDAR grid generator,
camera generator, and separate camera `add_pts_in_boxes` postprocessor. Their
geometry already existed in CMT; no UniBEV model/framework was imported. The
camera writer now includes its previously omitted point field directly.

Run `python -m unittest discover -s tests -v` with NumPy 1.x for synthetic geometry, map, CLI,
and filesystem regressions. The lightweight suite stubs SDK/framework interfaces
and uses temporary fixtures. With the SDK dependencies installed, enable the
optional real-SDK tests with:

```bash
ASYNCBEV_RUN_SDK_TESTS=1 python -m unittest discover -s tests -p test_flow_sdk.py -v
```

These tests construct temporary nuScenes metadata and compare serial/parallel
generation, numeric scene/pose motion, and repair file preservation. They never
read an existing dataset. Passing them does not establish whole-dataset
equivalence, GPU training/evaluation, or reproduction of the paper's results.
Do not replace a published dataset solely because a generator completed.
