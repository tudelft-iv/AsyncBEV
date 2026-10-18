"""Metadata-only generation and non-destructive repair of AsyncBEV flow labels.

Map and label pickle files must come from a trusted source. These tools preserve
the CMT label schema and use the shared CMT geometry without importing its model
plugin. No measured point clouds or camera images are read.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import os
import math
from pathlib import Path
import pickle
import re
import tempfile

import numpy as np


_TOKEN = re.compile(r"^[A-Za-z0-9_-]+$")
_FLOW_RTOL = 1e-5
_FLOW_ATOL = 1e-6
# Match the historical Python 3.8 writers' default and older NumPy readers.
_LABEL_PICKLE_PROTOCOL = 4


def points_for_sample(nusc, sync_data_token, grid):
    """Lazily load the metadata-only CMT point-selection implementation."""
    from projects.CMT.flow_geometry import points_for_sample as implementation
    return implementation(nusc, sync_data_token, grid)


def calculate_nusc_sceneflow(sync_data, async_data, points, nusc):
    """Lazily load CMT geometry without importing its detector plugin."""
    from projects.CMT.flow_geometry import calculate_nusc_sceneflow as implementation
    return implementation(sync_data, async_data, points, nusc)


def _token(value, name):
    if not isinstance(value, str) or not _TOKEN.fullmatch(value):
        raise ValueError("{} must be a safe, nonempty token: {!r}".format(name, value))
    return value


@dataclass(frozen=True)
class FlowTask:
    """One synchronized-sample / asynchronous-observation label."""

    sample_token: str
    sync_data_token: str
    async_data_token: str
    sync_cam_data_token: str = None

    def __post_init__(self):
        for name in ("sample_token", "sync_data_token", "async_data_token"):
            _token(getattr(self, name), name)
        if self.sync_cam_data_token is not None:
            _token(self.sync_cam_data_token, "sync_cam_data_token")

    @property
    def relative_path(self):
        return Path(self.sample_token) / (self.async_data_token + ".pkl")


def load_pickle(path):
    """Load a trusted local map or label pickle."""
    with open(path, "rb") as stream:
        return pickle.load(stream)


def _relative_candidates(value):
    """Strip a legacy root only at a full nuScenes samples/sweeps boundary."""
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("Map paths must be strings or path objects")
    parts = os.fspath(value).replace("\\", "/").split("/")
    if ".." in parts or "\x00" in os.fspath(value):
        raise ValueError("Unsafe map path: {!r}".format(value))
    return {
        "/".join(parts[index:])
        for index, part in enumerate(parts)
        if part in ("samples", "sweeps")
        and len(parts[index:]) >= 3
        and all(parts[index:])
    }


def _canonical_metadata_path(value):
    candidates = _relative_candidates(value)
    if len(candidates) != 1:
        raise ValueError("Ambiguous or invalid nuScenes filename: {!r}".format(value))
    return candidates.pop()


def _get(nusc, table, token):
    _token(token, table + " token")
    try:
        record = nusc.get(table, token)
    except (KeyError, ValueError) as exc:
        raise ValueError("Unknown {} token {!r}".format(table, token)) from exc
    if record.get("token", token) != token:
        raise ValueError("Metadata token mismatch for {} {!r}".format(table, token))
    return record


def _channel(nusc, sample_data):
    channel = sample_data.get("channel")
    if channel is None:
        calibrated = _get(nusc, "calibrated_sensor", sample_data["calibrated_sensor_token"])
        channel = _get(nusc, "sensor", calibrated["sensor_token"])["channel"]
    return channel


def _require_channel(nusc, sample_data, expected):
    if _channel(nusc, sample_data) != expected:
        raise ValueError("Observation {!r} is not {}".format(sample_data.get("token"), expected))


def _check_frame_path(frame, key, observation):
    if key in frame:
        expected = _canonical_metadata_path(observation["filename"])
        if expected not in _relative_candidates(frame[key]):
            raise ValueError("{} does not match asynchronous observation filename".format(key))


def _timestamp(record):
    value = record.get("timestamp")
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)) or not math.isfinite(value):
        raise ValueError("Observation timestamp must be a finite number")
    return value


def collect_tasks(nusc, mapping, modality, sample_tokens=None, limit=None):
    """Resolve legacy map paths to checked, deterministic generation tasks.

    Camera maps contain all cameras; only CAM_BACK_LEFT supplies the historical
    camera-flow target used by CMT. Matching never falls back to a basename.
    Filtering and the task limit are applied after validating the relevant map.
    """
    if modality not in ("camera", "lidar"):
        raise ValueError("modality must be 'camera' or 'lidar'")
    if not isinstance(mapping, Mapping):
        raise ValueError("The asynchronous map must be a dictionary")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be a positive integer")
    channel = "CAM_BACK_LEFT" if modality == "camera" else "LIDAR_TOP"
    samples = {sample["token"]: sample for sample in nusc.sample}
    selected = None if sample_tokens is None else set(sample_tokens)
    if selected is not None:
        for token in selected:
            _token(token, "selected sample token")
        unknown = selected.difference(samples)
        if unknown:
            raise ValueError("Unknown selected sample tokens: {}".format(sorted(unknown)))
    index = {}
    for sample in samples.values():
        anchor = _get(nusc, "sample_data", sample["data"][channel])
        _require_channel(nusc, anchor, channel)
        if anchor["sample_token"] != sample["token"]:
            raise ValueError("Synchronized observation has the wrong sample token")
        path = _canonical_metadata_path(anchor["filename"])
        if path in index:
            raise ValueError("Ambiguous synchronized observation path: " + path)
        index[path] = sample

    tasks = []
    seen = set()
    for key, frames in mapping.items():
        candidates = _relative_candidates(key)
        # A camera map includes five channels that are intentionally unused.
        channels = {candidate.split("/")[1] for candidate in candidates}
        if modality == "camera" and channels and all(
            value.startswith("CAM_") and value != channel for value in channels
        ):
            continue
        matches = candidates.intersection(index)
        if len(matches) != 1:
            raise ValueError("Unknown or ambiguous {} map path: {!r}".format(modality, key))
        sample = index[matches.pop()]
        anchor_timestamp = _timestamp(_get(nusc, "sample_data", sample["data"][channel]))
        sync_token = sample["data"]["LIDAR_TOP"]
        sync_record = _get(nusc, "sample_data", sync_token)
        _require_channel(nusc, sync_record, "LIDAR_TOP")
        if sync_record["sample_token"] != sample["token"]:
            raise ValueError("Synchronized LiDAR has the wrong sample token")
        if not isinstance(frames, (list, tuple)):
            raise ValueError("Map frames must be a list or tuple for {!r}".format(key))
        for frame in frames:
            if not isinstance(frame, Mapping):
                raise ValueError("Each asynchronous map frame must be a dictionary")
            token_key = "data_token" if modality == "camera" else "sample_token"
            if token_key not in frame:
                raise ValueError("Map frame is missing {}".format(token_key))
            async_token = frame[token_key]
            observation = _get(nusc, "sample_data", async_token)
            _require_channel(nusc, observation, channel)
            if _timestamp(observation) > anchor_timestamp:
                raise ValueError("Asynchronous observation is newer than its synchronized channel anchor")
            async_sample = samples.get(observation["sample_token"])
            if async_sample is None:
                async_sample = _get(nusc, "sample", observation["sample_token"])
            if async_sample["scene_token"] != sample["scene_token"]:
                raise ValueError("Asynchronous observation belongs to a different scene")
            if "timestamp" in frame and frame["timestamp"] != observation["timestamp"]:
                raise ValueError("Map timestamp does not match asynchronous observation")
            if modality == "camera" and "sample_token" in frame:
                # Historical sweep rows may identify an anchor sample, while
                # sample_data belongs to the following keyframe. Both must be
                # in this scene; keyframe rows must identify the exact sample.
                row_sample = samples.get(frame["sample_token"])
                if row_sample is None:
                    row_sample = _get(nusc, "sample", frame["sample_token"])
                if row_sample["scene_token"] != sample["scene_token"]:
                    raise ValueError("Camera map row sample belongs to a different scene")
                if observation.get("is_key_frame") and frame["sample_token"] != observation["sample_token"]:
                    raise ValueError("Camera keyframe row has the wrong sample token")
            _check_frame_path(frame, "data_path" if modality == "camera" else "lidar_path", observation)
            task = FlowTask(
                sample["token"], sync_token, async_token,
                sample["data"]["CAM_BACK_LEFT"] if modality == "camera" else None,
            )
            if task.relative_path in seen:
                raise ValueError("Duplicate output task: {}".format(task.relative_path))
            seen.add(task.relative_path)
            if selected is None or task.sample_token in selected:
                tasks.append(task)
    tasks.sort(key=lambda task: (task.sample_token, task.async_data_token))
    return tasks if limit is None else tasks[:limit]


def _array(value, name):
    if not isinstance(value, np.ndarray) or value.ndim != 2 or value.shape[1] != 3:
        raise ValueError("{} must be an N x 3 NumPy array".format(name))
    if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
        raise ValueError("{} must contain finite real numbers".format(name))
    return value


def _tokens_for_task(task):
    if task.sync_cam_data_token is not None:
        return {
            "sync_lidar_data_token": task.sync_data_token,
            "sync_cam_data_token": task.sync_cam_data_token,
            "async_cam_data_token": task.async_data_token,
        }
    return {"sync_data_token": task.sync_data_token, "async_data_token": task.async_data_token}


def _validate_payload(payload, task, require_points=True):
    if not isinstance(payload, dict):
        raise ValueError("Flow label must be a dictionary")
    for key, expected in _tokens_for_task(task).items():
        if payload.get(key) != expected:
            raise ValueError("Flow label {} does not match task token".format(key))
    if "gt_flow_sync2async" not in payload:
        raise ValueError("Flow label is missing gt_flow_sync2async")
    scene_flow = _array(payload["gt_flow_sync2async"], "gt_flow_sync2async")
    pose_keys = [key for key in ("pose_flow_sync2async", "pose_flow_s2t") if key in payload]
    if not pose_keys:
        raise ValueError("Flow label is missing pose flow")
    pose = _array(payload[pose_keys[0]], pose_keys[0])
    if len(pose_keys) == 2:
        alternate = _array(payload[pose_keys[1]], pose_keys[1])
        if not np.array_equal(pose, alternate):
            raise ValueError("The two pose-flow fields disagree")
    if pose.shape != scene_flow.shape:
        raise ValueError("Scene and pose flow shapes disagree")
    if require_points and "pts_in_boxes" not in payload:
        raise ValueError("Flow label is missing pts_in_boxes")
    if "pts_in_boxes" in payload:
        points = _array(payload["pts_in_boxes"], "pts_in_boxes")
        if points.shape != scene_flow.shape:
            raise ValueError("Point and flow shapes disagree")
    return scene_flow, pose


def _require_agreement(actual, expected, name):
    if actual.shape != expected.shape or not np.allclose(
        actual, expected, rtol=_FLOW_RTOL, atol=_FLOW_ATOL
    ):
        raise ValueError("{} does not match the current metadata/grid geometry".format(name))


def _validate_geometry(payload, task, points, scene_flow, pose_flow, require_points=True):
    saved_scene, saved_pose = _validate_payload(payload, task, require_points)
    _require_agreement(saved_scene, scene_flow, "Scene flow")
    _require_agreement(saved_pose, pose_flow, "Pose flow")
    if "pts_in_boxes" in payload:
        _require_agreement(payload["pts_in_boxes"], points, "Point coordinates")


def _contained_path(root, relative):
    root = Path(root).expanduser().resolve()
    path = root / relative
    if path.is_symlink():
        raise ValueError("Refusing a symbolic-link label path: {}".format(path))
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError("Label path escapes its root: {}".format(path)) from exc
    return path


def _separate_roots(input_dir, output_dir):
    source = Path(input_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Repair input and output directories must not overlap")


def _same_payload(left, right):
    """Check every preserved field before accepting an existing repair output."""
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
                and left.dtype == right.dtype and np.array_equal(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same_payload(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and type(left) is type(right):
        return len(left) == len(right) and all(_same_payload(a, b) for a, b in zip(left, right))
    try:
        return type(left) is type(right) and bool(left == right)
    except (ValueError, TypeError):
        return False


def _publish_bytes(path, content):
    """Publish complete bytes atomically, refusing to replace any existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".async-flow-", suffix=".tmp", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        # A hard link is an atomic create-if-absent operation on the same
        # filesystem. Unlike os.replace, this never clobbers a competing writer.
        os.link(str(temporary), str(path))
    finally:
        if temporary is not None:
            temporary.unlink()


def process_task(task, nusc, grid, mode, output_dir=None, input_dir=None,
                 dry_run=False, existing="error"):
    """Generate a label or copy/repair it into a separate tree.

    Repair always checks existing scene/pose flow against recomputed geometry,
    but only the missing coordinate field is added. Existing arrays and extra
    metadata are preserved. A dry run never creates directories or label files.
    """
    if mode not in ("generate", "repair"):
        raise ValueError("mode must be 'generate' or 'repair'")
    if existing not in ("error", "validate"):
        raise ValueError("existing must be 'error' or 'validate'")
    if mode == "generate" and output_dir is None:
        raise ValueError("Generation requires output_dir")
    if mode == "repair" and input_dir is None:
        raise ValueError("Repair requires input_dir")
    if mode == "repair" and not dry_run and output_dir is None:
        raise ValueError("Applying repair requires a separate output_dir")
    if mode == "repair" and output_dir is not None:
        _separate_roots(input_dir, output_dir)
    destination = None if output_dir is None else _contained_path(output_dir, task.relative_path)
    destination_exists = destination is not None and destination.exists()
    if destination_exists and existing == "error":
        raise FileExistsError("Output already exists: {}".format(destination))

    source_bytes = None
    payload = None
    had_points = False
    if mode == "repair":
        source = _contained_path(input_dir, task.relative_path)
        source_bytes = source.read_bytes()
        payload = pickle.loads(source_bytes)
        _validate_payload(payload, task, require_points=False)
        had_points = "pts_in_boxes" in payload

    points = _array(points_for_sample(nusc, task.sync_data_token, grid), "Reconstructed points")
    scene_flow, pose_flow = calculate_nusc_sceneflow(
        _get(nusc, "sample_data", task.sync_data_token),
        _get(nusc, "sample_data", task.async_data_token), points, nusc,
    )
    _array(scene_flow, "Recomputed scene flow")
    _array(pose_flow, "Recomputed pose flow")
    if scene_flow.shape != points.shape or pose_flow.shape != points.shape:
        raise ValueError("Recomputed point and flow shapes disagree")
    if mode == "generate":
        payload = dict(_tokens_for_task(task), pts_in_boxes=points, gt_flow_sync2async=scene_flow)
        pose_key = "pose_flow_s2t" if task.sync_cam_data_token is not None else "pose_flow_sync2async"
        payload[pose_key] = pose_flow
        _validate_payload(payload, task)
    else:
        _validate_geometry(payload, task, points, scene_flow, pose_flow, require_points=False)
        if not had_points:
            payload["pts_in_boxes"] = points

    if destination_exists:
        saved = load_pickle(destination)
        _validate_geometry(saved, task, points, scene_flow, pose_flow)
        if mode == "repair" and not _same_payload(saved, payload):
            raise ValueError("Existing output differs from the repaired source payload")
        return {"status": "validated", "path": str(destination)}
    path = str(destination) if destination is not None else str(source)
    if dry_run:
        status = "would_generate" if mode == "generate" else ("already_complete" if had_points else "would_repair")
        return {"status": status, "path": path}
    if (mode == "generate" or not had_points) and int(np.__version__.split('.')[0]) >= 2:
        raise RuntimeError("Writing CMT-compatible label pickles requires NumPy 1.x; "
                           "NumPy 2.x pickles are not readable by legacy CMT. "
                           "Use a separate environment with numpy<2.")
    content = source_bytes if mode == "repair" and had_points else pickle.dumps(payload, protocol=_LABEL_PICKLE_PROTOCOL)
    _publish_bytes(destination, content)
    status = "generated" if mode == "generate" else ("already_complete" if had_points else "repaired")
    return {"status": status, "path": path}
