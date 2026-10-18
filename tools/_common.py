"""Shared configuration imports for the MMCV 1.x command-line tools."""

import importlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def import_plugins(cfg, config_path):
    """Register custom modules and a directory-based CMT plugin.

    Repository-relative plugin paths do not require a trailing slash. Absolute
    paths are also accepted; an external plugin must be an importable package.
    Data and checkpoint paths retain the normal working-directory semantics.
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    if cfg.get('custom_imports'):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    if not cfg.get('plugin', False):
        return

    plugin_dir = cfg.get('plugin_dir')
    if plugin_dir is None:
        plugin_path = Path(config_path).expanduser().resolve().parent
    else:
        plugin_path = Path(plugin_dir).expanduser()
        if not plugin_path.is_absolute():
            plugin_path = REPO_ROOT / plugin_path
        plugin_path = plugin_path.resolve()
    if not plugin_path.is_dir():
        raise FileNotFoundError('Plugin directory does not exist: {}'.format(
            plugin_path))

    try:
        module_parts = plugin_path.relative_to(REPO_ROOT).parts
    except ValueError:
        # Resolve a conventional external package from its package root.
        package_root = plugin_path
        while (package_root / '__init__.py').is_file():
            package_root = package_root.parent
        if package_root == plugin_path:
            raise ValueError('External plugin_dir must contain __init__.py: '
                             '{}'.format(plugin_path))
        sys.path.insert(0, str(package_root))
        module_parts = plugin_path.relative_to(package_root).parts
    if not module_parts:
        raise ValueError('plugin_dir must point to a package, not the '
                         'repository root')
    importlib.import_module('.'.join(module_parts))
