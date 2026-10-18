"""Shared CPU-only CLI orchestration; --help needs only the standard library."""
import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import time


_WORKER = None


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return parsed


def parse_args(mode, argv=None):
    parser = argparse.ArgumentParser(
        description=('Generate camera/LiDAR GT grid flow into new files.'
                     if mode == 'generate' else
                     'Check legacy labels and reconstruct pts_in_boxes. '
                     'Read-only by default; --apply writes separate copies.'))
    parser.add_argument('--data-root', required=True,
                        help='nuScenes root containing versioned JSON metadata')
    parser.add_argument('--map-file', required=True,
                        help='matching trusted AsyncBEV camera/LiDAR map PKL')
    parser.add_argument('--version', default='v1.0-trainval',
                        choices=['v1.0-trainval', 'v1.0-mini'])
    parser.add_argument('--modality', choices=['camera', 'lidar'],
                        required=mode == 'generate',
                        default=None if mode == 'generate' else 'camera')
    parser.add_argument('--output-dir', required=mode == 'generate',
                        help='label root: <sample_token>/<async_token>.pkl')
    parser.add_argument('--workers', type=positive_int, default=1,
                        help='CPU processes (default: 1); each loads metadata/grid')
    parser.add_argument('--bev-size', type=float, nargs=2, default=[200, 200],
                        metavar=('H', 'W'), help='historical normalized grid size')
    parser.add_argument('--resolution', type=float, default=0.5,
                        help='historical grid discretization parameter, not '
                        'physical XY spacing in meters (default: 0.5)')
    parser.add_argument('--point-cloud-range', type=float, nargs=6,
                        default=[-54, -54, -5, 54, 54, 3],
                        metavar=('XMIN', 'YMIN', 'ZMIN', 'XMAX', 'YMAX', 'ZMAX'))
    parser.add_argument('--sample-token', action='append', dest='sample_tokens',
                        help='restrict to a nuScenes sample; may be repeated')
    parser.add_argument('--limit', type=positive_int,
                        help='process at most this many selected label pairs')
    parser.add_argument('--existing', choices=['error', 'validate'],
                        default='error', help='destination collision policy; '
                        'validate resumes compatible files, never overwrites')
    parser.add_argument('--report', help='optional new JSON report file')
    if mode == 'repair':
        parser.add_argument('--input-dir', required=True,
                            help='existing label root to read; never modified')
        operation = parser.add_mutually_exclusive_group()
        operation.add_argument('--apply', action='store_true',
                               help='write validated copies to --output-dir')
        operation.add_argument('--dry-run', action='store_true',
                               help='explicit read-only check (the default)')
    else:
        parser.add_argument('--dry-run', action='store_true',
                            help='compute and validate without writing labels')
    args = parser.parse_args(argv)
    if mode == 'repair':
        args.dry_run = not args.apply
        if args.apply and not args.output_dir:
            parser.error('--apply requires --output-dir; in-place repair is unsupported')
    values = args.bev_size + args.point_cloud_range + [args.resolution]
    if not all(math.isfinite(value) for value in values):
        parser.error('grid parameters must be finite')
    if args.resolution <= 0 or min(args.bev_size) <= 0:
        parser.error('--resolution and --bev-size must be positive')
    if any(args.point_cloud_range[i + 3] <= args.point_cloud_range[i]
           for i in range(3)):
        parser.error('point-cloud range maxima must exceed minima')
    return args


def _thread_defaults():
    # Respect caller choices, while avoiding many BLAS threads per worker.
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(name, '1')


def _initialize_worker(data_root, version, grid_options, options):
    global _WORKER
    _thread_defaults()
    from nuscenes.nuscenes import NuScenes
    from projects.CMT.flow_geometry import build_reference_grid
    _WORKER = (NuScenes(version=version, dataroot=data_root, verbose=False),
               build_reference_grid(**grid_options), options)


def _execute(task):
    from tools.data_converter.async_flow import process_task
    nusc, grid, options = _WORKER
    try:
        return process_task(task, nusc, grid, **options)
    except Exception as error:
        return dict(status='error', path=str(task.relative_path),
                    error='{}: {}'.format(type(error).__name__, error))


def _parallel_results(tasks, workers, initargs):
    # Bound submitted tasks so a full supplement does not enqueue 500k futures.
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context('spawn'),
                             initializer=_initialize_worker,
                             initargs=initargs) as pool:
        remaining = iter(tasks)
        pending = set()
        for _ in range(workers * 2):
            task = next(remaining, None)
            if task is not None:
                pending.add(pool.submit(_execute, task))
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()
                task = next(remaining, None)
                if task is not None:
                    pending.add(pool.submit(_execute, task))


def _overlap(first, second):
    return first == second or first in second.parents or second in first.parents


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _write_report(path, report):
    """Publish a complete report exclusively; never clobber an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                         dir=str(path.parent), delete=False) as f:
            temporary = Path(f.name)
            json.dump(report, f, indent=2, sort_keys=True)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.link(str(temporary), str(path))
    finally:
        if temporary is not None:
            temporary.unlink()


def main(mode, argv=None):
    global _WORKER
    args = parse_args(mode, argv)
    started = time.monotonic()
    report = dict(mode=mode, parameters=vars(args).copy(), counts={},
                  errors=[], selected=0, completed=0)
    report_path = None
    try:
        root = Path(args.data_root).expanduser().resolve()
        map_path = Path(args.map_file).expanduser().resolve()
        output = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
        source = (Path(args.input_dir).expanduser().resolve()
                  if mode == 'repair' else None)
        if not root.is_dir():
            raise ValueError('--data-root must be an existing nuScenes root')
        if not map_path.is_file():
            raise ValueError('--map-file must be an existing PKL')
        if source is not None:
            if not source.is_dir():
                raise ValueError('--input-dir must be an existing label directory')
            if output is not None and _overlap(source, output):
                raise ValueError('--output-dir must be separate from and not '
                                 'overlap --input-dir')
        if args.report:
            candidate = Path(args.report).expanduser().resolve()
            if candidate.exists():
                raise FileExistsError('report already exists: {}'.format(candidate))
            if source is not None and _overlap(source, candidate):
                raise ValueError('--report must be outside the input label directory')
            report_path = candidate
        _thread_defaults()
        from nuscenes.nuscenes import NuScenes
        from projects.CMT.flow_geometry import build_reference_grid
        from tools.data_converter.async_flow import collect_tasks, load_pickle
        report['python_version'] = sys.version
        report['geometry_sha256'] = _file_sha256(
            Path(__file__).resolve().parents[2] / 'projects/CMT/flow_geometry.py')
        try:
            from importlib.metadata import version, PackageNotFoundError
            report['dependencies'] = {}
            for package in ('numpy', 'nuscenes-devkit', 'pyquaternion'):
                try:
                    report['dependencies'][package] = version(package)
                except PackageNotFoundError:
                    report['dependencies'][package] = 'unavailable'
        except ImportError:
            pass
        nusc = NuScenes(version=args.version, dataroot=str(root), verbose=False)
        mapping = load_pickle(map_path)
        tasks = collect_tasks(nusc, mapping, args.modality,
                              sample_tokens=args.sample_tokens, limit=args.limit)
        if not tasks:
            raise ValueError('selection contains no label pairs; check map/version/tokens')
        report['selected'] = len(tasks)
        report['map_sha256'] = _file_sha256(map_path)
        report['resolved_paths'] = dict(data_root=str(root), map_file=str(map_path),
                                        input_dir=str(source) if source else None,
                                        output_dir=str(output) if output else None)
        grid_options = dict(bev_size=args.bev_size, resolution=args.resolution,
                            point_cloud_range=args.point_cloud_range)
        options = dict(mode=mode, input_dir=source, output_dir=output,
                       dry_run=args.dry_run, existing=args.existing)
        if args.workers == 1:
            _WORKER = (nusc, build_reference_grid(**grid_options), options)
            results = map(_execute, tasks)
        else:
            # Validate the grid before dispatch; each worker owns its own copy.
            build_reference_grid(**grid_options)
            results = _parallel_results(
                tasks, args.workers, (str(root), args.version, grid_options, options))
        counts = Counter()
        for result in results:
            counts[result['status']] += 1
            report['counts'] = dict(counts)
            report['completed'] += 1
            if result['status'] == 'error' and len(report['errors']) < 100:
                report['errors'].append(result)
                print('{}: {}'.format(result['path'], result['error']), file=sys.stderr)
            if report['completed'] % 100 == 0:
                print('{}/{} labels checked'.format(report['completed'], len(tasks)),
                      file=sys.stderr, flush=True)
        report['ok'] = not counts['error']
    except Exception as error:
        report['ok'] = False
        report['fatal_error'] = '{}: {}'.format(type(error).__name__, error)
        print(report['fatal_error'], file=sys.stderr)
        if isinstance(error, ImportError):
            print('These CPU tools require NumPy, nuscenes-devkit and pyquaternion; '
                  'see tools/data_converter/README.md.', file=sys.stderr)
    finally:
        _WORKER = None
    report['elapsed_seconds'] = round(time.monotonic() - started, 3)
    if report_path is not None:
        try:
            _write_report(report_path, report)
        except OSError as error:
            report['ok'] = False
            report['report_error'] = str(error)
            print('Could not write report: {}'.format(error), file=sys.stderr)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report['ok'] else 1
