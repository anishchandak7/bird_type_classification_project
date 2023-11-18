"""Module contains common functionalities to be utilzed in the project"""
# Necessary imports
# standard imports
import os
# Third-party imports
from pathlib import Path
from ensure import ensure_annotations
from box.exceptions import BoxValueError
from box import ConfigBox
import yaml
# Local application/library specific imports
from cnnClassifier import logger


@ensure_annotations
def read_yaml(yaml_file_path: Path) -> ConfigBox:
    """Read a yaml file and return it as a dictionary object using the 'box' library."""
    try:
        with open(yaml_file_path, encoding='utf-8') as yaml_file:
            yaml_file_content = yaml.safe_load(yaml_file)
            logger.info("%s loaded successfully.", yaml_file_path)
    except BoxValueError as exc:
        raise ValueError("The provided YAML file is empty") from exc
    except Exception as e:
        raise e
    return ConfigBox(yaml_file_content)

@ensure_annotations
def create_directories(directories_path: list, verbose = True):
    """Create list of directories if they don't exist already."""

    for dir_path in directories_path:
        os.makedirs(dir_path, exist_ok=True)
        if verbose:
            logger.info("Created directory at: %s", dir_path)
