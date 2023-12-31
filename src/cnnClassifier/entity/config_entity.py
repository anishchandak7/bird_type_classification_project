"""This file contains the dataclasses for each step"""
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    """
    This class will help in reading configurations from config.yaml file 
    for data ingestion step.
    """
    root_dir: Path
    source_url: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    """
    This class will help in reading configurations from config.yaml file for base model preparation.
    """
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int
    params_acceptable_classes: list
    params_pooling: str

@dataclass(frozen=True)
class PrepareCallbacksConfig:
    """
    This class will help in reading configurations from config.yaml file for callbacks.
    """
    root_dir: Path
    tensorboard_root_log_dir: Path
    checkpoint_model_filepath: Path

@dataclass(frozen=True)
class TrainingConfig:
    """
    This class will help in reading configurations from config.yaml file for training.
    """
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    validation_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list
    params_acceptable_classes: list

@dataclass(frozen=True)
class EvaluationConfig:
    """
    This class will help in reading configurations from config.yaml and params.yaml files for evaluation.
    """
    path_of_model: Path
    training_data: Path
    test_data:Path
    all_params: dict
    params_image_size: list
    params_batch_size: int
    params_acceptable_classes:list

@dataclass(frozen=True)
class PredictionConfig:
    """
    This class will help in reading configurations from config.yaml and params.yaml files for prediction.
    """
    params_acceptable_classes:list
