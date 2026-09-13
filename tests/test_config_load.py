from pathlib import Path

import pytest


def test_project_root_and_config_paths():
    from utils.population_io import get_project_root, get_config_path

    root = get_project_root()
    assert root.is_dir(), f"Project root should be a directory: {root}"
    config_dir = get_config_path()
    assert config_dir == root / "config"
    assert (root / "config" / "RGConfig.yaml").exists(), "RGConfig.yaml should exist"
    assert (root / "config" / "PGConfig.yaml").exists(), "PGConfig.yaml should exist"


def test_rg_config_yaml_loadable():
    from utils.population_io import get_project_root
    import yaml

    path = get_project_root() / "config" / "RGConfig.yaml"
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    assert data is not None and isinstance(data, dict), "RGConfig should be a non-empty dict"


def test_pg_config_yaml_loadable():
    from utils.population_io import get_project_root
    import yaml

    path = get_project_root() / "config" / "PGConfig.yaml"
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    assert data is not None
    assert isinstance(data, dict), "PGConfig should be a dict"
