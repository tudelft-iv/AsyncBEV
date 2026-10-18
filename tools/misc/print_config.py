# Copyright (c) OpenMMLab. All rights reserved.
"""Inspect the resolved MMCV configuration without building a model."""
import argparse


def parse_args():
    # Imported here so importing this utility does not initialize the plugin.
    from mmcv import DictAction

    parser = argparse.ArgumentParser(
        description='Print a resolved MMCV config, including inherited values.',
        epilog='Overrides are applied after Python config execution. Changing '
        'a top-level variable does not reconstruct nested pipeline settings.')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--cfg-options', '--options', dest='cfg_options', nargs='+',
        action=DictAction, help='override resolved values with key=value pairs')
    parser.add_argument('--out', help='optionally save the resolved config')
    return parser.parse_args()


def main():
    from mmcv import Config

    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    print(cfg.pretty_text)
    if args.out:
        cfg.dump(args.out)


if __name__ == '__main__':
    main()
