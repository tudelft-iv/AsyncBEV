# Copyright (c) OpenMMLab. All rights reserved.
"""Analyze MMCV JSON-lines logs without loading the detector or CUDA.

Adapted from MMDetection3D's analysis tool. Rows remain intact so training and
validation metrics cannot become misaligned when their logged keys differ.
"""
import argparse
import json
import math
from numbers import Real
import statistics


def load_json_logs(json_logs):
    """Group full log records by their actual epoch, preserving file order."""
    log_dicts = []
    for json_log in json_logs:
        epochs = {}
        with open(json_log, encoding='utf-8') as log_file:
            for line_no, line in enumerate(log_file, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except ValueError as exc:
                    raise ValueError(f'{json_log}:{line_no}: invalid JSON') from exc
                if not isinstance(record, dict):
                    raise ValueError(f'{json_log}:{line_no}: expected a JSON object')
                if 'epoch' not in record:  # MMCV's environment/config header.
                    continue
                epoch = record['epoch']
                if isinstance(epoch, bool) or not isinstance(epoch, int):
                    raise ValueError(f'{json_log}:{line_no}: epoch must be an integer')
                epochs.setdefault(epoch, []).append(record)
        log_dicts.append(epochs)
    return log_dicts


def is_number(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def cal_train_time(log_dicts, args):
    for json_log, epochs in zip(args.json_logs, log_dicts):
        print(f'Analyze train time of {json_log}')
        epoch_times = {}
        for epoch, records in epochs.items():
            times = [row['time'] for row in records
                     if row.get('mode', 'train') == 'train'
                     and is_number(row.get('time')) and row['time'] >= 0]
            if not args.include_outliers:
                times = times[1:]
            if times:
                epoch_times[epoch] = times
        if not epoch_times:
            raise ValueError(f'{json_log}: no training timing samples remain; '
                             'try --include-outliers for short logs')
        means = {epoch: statistics.mean(times)
                 for epoch, times in epoch_times.items()}
        slowest = max(means, key=means.get)
        fastest = min(means, key=means.get)
        print(f'slowest epoch {slowest}, average time is {means[slowest]:.4f}')
        print(f'fastest epoch {fastest}, average time is {means[fastest]:.4f}')
        print(f'time std over epochs is {statistics.pstdev(means.values()):.4f}')
        all_times = [value for times in epoch_times.values() for value in times]
        print(f'average iter time: {statistics.mean(all_times):.4f} s/iter '
              f'({len(all_times)} logged timing samples)')
        print()


def metric_points(epochs, metric, mode, interval):
    """Use real epochs for validation and log-record indices for training.

    Training log indices avoid inventing global iterations for resumed or
    incomplete epochs, since MMCV logs only contain within-epoch iterations.
    """
    xs, ys = [], []
    training_index = 0
    for epoch, records in epochs.items():
        for row in records:
            record_mode = row.get('mode', 'train')
            if record_mode == 'train':
                training_index += 1
            wanted_mode = 'val' if mode == 'eval' else 'train'
            if record_mode != wanted_mode or epoch % interval:
                continue
            if not is_number(row.get(metric)):
                continue
            xs.append(epoch if mode == 'eval' else training_index)
            ys.append(row[metric])
    return xs, ys


def plot_curve(log_dicts, args):
    # Timing analysis requires only Python's standard library. Plotting has
    # optional dependencies, imported only for this command.
    import matplotlib
    if args.backend:
        matplotlib.use(args.backend)
    from matplotlib import pyplot as plt
    if args.style:
        import seaborn as sns
        sns.set_style(args.style)

    legend = args.legend or [f'{path}: {key}'
                             for path in args.json_logs for key in args.keys]
    if len(legend) != len(args.json_logs) * len(args.keys):
        raise ValueError('--legend must have one label per log/metric pair')
    fig, ax = plt.subplots()
    try:
        for index, (path, epochs) in enumerate(zip(args.json_logs, log_dicts)):
            for metric_index, metric in enumerate(args.keys):
                xs, ys = metric_points(epochs, metric, args.mode, args.interval)
                if not xs:
                    raise ValueError(f'{path}: no {args.mode} values for {metric!r} '
                                     'at the selected epoch interval')
                label = legend[index * len(args.keys) + metric_index]
                ax.plot(xs, ys, label=label,
                        marker='o' if args.mode == 'eval' else None,
                        linewidth=1)
        ax.set_xlabel('epoch' if args.mode == 'eval' else 'logged training record')
        if args.title:
            ax.set_title(args.title)
        ax.legend()
        fig.tight_layout()
        if args.out:
            fig.savefig(args.out)
            print(f'Saved curve to {args.out}')
        else:
            plt.show()
    finally:
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='task', required=True)
    time_parser = subparsers.add_parser('cal_train_time', help='summarize training time')
    time_parser.add_argument('json_logs', nargs='+', help='MMCV JSON-lines log files')
    time_parser.add_argument(
        '--include-outliers', action='store_true',
        help='include the first logged timing sample of each epoch')
    plot_parser = subparsers.add_parser('plot_curve', help='plot training or validation metrics')
    plot_parser.add_argument('json_logs', nargs='+', help='MMCV JSON-lines log files')
    plot_parser.add_argument('--keys', nargs='+', default=['loss'], help='exact logged metric keys')
    plot_parser.add_argument('--title')
    plot_parser.add_argument('--legend', nargs='+')
    plot_parser.add_argument('--backend', help='Matplotlib backend, e.g. Agg for headless use')
    plot_parser.add_argument('--style', default=None, help='optional seaborn style')
    plot_parser.add_argument('--out', help='output image; otherwise display interactively')
    plot_parser.add_argument('--mode', choices=['train', 'eval'], default='train')
    plot_parser.add_argument(
        '--interval', type=int, default=1,
        help='select epochs divisible by this value (default: all epochs)')
    args = parser.parse_args()
    if getattr(args, 'interval', 1) < 1:
        parser.error('--interval must be positive')
    return args


def main():
    args = parse_args()
    try:
        log_dicts = load_json_logs(args.json_logs)
        tasks = {'cal_train_time': cal_train_time, 'plot_curve': plot_curve}
        tasks[args.task](log_dicts, args)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == '__main__':
    main()
