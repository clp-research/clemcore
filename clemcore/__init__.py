import logging.config
import os

import yaml
import importlib.resources as importlib_resources
from importlib.metadata import version, PackageNotFoundError

import clemcore.backends as backends


def get_version():
    try:
        return version("clemcore")
    except PackageNotFoundError:
        return "unknown"


def load_logging_config():
    pkg_file_path = "utils/logging.yaml"
    with importlib_resources.files(__package__).joinpath(pkg_file_path).open("r") as f:
        default_config = yaml.safe_load(f)

    custom_file = os.path.join(os.getcwd(), "logging.yaml")
    if os.path.exists(custom_file):
        with open(custom_file) as f:
            custom_config = yaml.safe_load(f)
        return {**default_config, **custom_config}
    return default_config


try:
    import logging

    logging.config.dictConfig(load_logging_config())
except Exception as e:
    print(f"Failed to load logging config: {e}")
