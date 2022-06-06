from pathlib import Path
import os
import yaml


def get_project_root() -> Path:
    return Path(__file__).parent


def read_config_file():
    root_dir = get_project_root()
    config_path = os.path.join(root_dir, 'user_params.yaml')
    return yaml.safe_load(open(config_path))

