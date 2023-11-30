"""Module contains common functionalities to be utilzed in the project"""
# Necessary imports
# standard imports
import os
# Third-party imports
import base64
from pathlib import Path
import random
import json
from ensure import ensure_annotations
from box.exceptions import BoxValueError
from box import ConfigBox
import yaml
import pandas as pd
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

@ensure_annotations
def generate_dataframe(folder_path: Path|str, classes:list, verbose = False):
    """
    Generate a dataframe for ingested data.
    Args:
        folder_path (Path|str): ingested data folder path
        classes (list): A list of classes for which data must be present.
        verbose (bool, optional): If True, then for each class, randomly pick 20% of data. Defaults to False.
    Returns:
        Pandas DataFrame: Return a dataframe with columns filepath, label. 
    """

    # Since each subfolder name is a label, hence we need a list of labels.
    labels_list = os.listdir(folder_path)
    file_paths = []
    labels = []
    for label in labels_list:

        label_folder_path = os.path.join(folder_path, label)
        label_files_list = os.listdir(label_folder_path)

        temp_file_paths = []
        temp_labels = []

        for label_file in label_files_list:
            file_path = os.path.join(label_folder_path, label_file).replace('\\','/')
            temp_file_paths.append(file_path)
            temp_labels.append(label)
        
        if verbose:
            n = int(0.20 * len(temp_file_paths))
            file_paths.extend(random.sample(temp_file_paths, n))
            labels.extend(random.sample(temp_labels, n))
        else:
            file_paths.extend(temp_file_paths)
            labels.extend(temp_labels)

    df = pd.DataFrame({'filepath': file_paths, 'label': labels})

    df = df[df['label'].isin(classes)]

    return df

@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=4)

    logger.info("json file saved at: %s", path)


def decode_image(img_string, file_name):
    """
    decode image from base64 string

    Args:
        img_string (_type_): _description_
        file_name (_type_): _description_
    """
    img_data = base64.b64decode(img_string)
    with open(file_name, 'wb') as f:
        f.write(img_data)


def encode_image_into_base64(cropped_image_path):
    """
    encode image into base64 string

    Args:
        cropped_image_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    with open(cropped_image_path, "rb") as f:
        return base64.b64encode(f.read())