"""This file contains ConfigurationManager Class"""
import os
from pathlib import Path
from cnnClassifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import (DataIngestionConfig,
                                                PrepareBaseModelConfig,
                                                PrepareCallbacksConfig)

class ConfigurationManager:
    """class which manages configurations for all the steps of the project."""
    def __init__(self, 
                 config_file_path = CONFIG_FILE_PATH,
                 params_file_path = PARAMS_FILE_PATH) -> None:       
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Encasulates data ingestion configurations from config.yaml file
        and return as DataIngestionConfig object.
        """
        data_ingestion_config = self.config.data_ingestion

        # Create data ingestion folder inside artifacts.
        create_directories([data_ingestion_config.root_dir])

        # Encasulates the configurations and return it as DataIngestionConfig object.
        return DataIngestionConfig(root_dir=data_ingestion_config.root_dir,
                            source_url=data_ingestion_config.source_url,
                            local_data_file=data_ingestion_config.local_data_file,
                            unzip_dir=data_ingestion_config.unzip_dir)

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        """
        Encasulates base model configurations from config.yaml and params.yaml file.

        Returns:
            PrepareBaseModelConfig Object
        """
        config = self.config.prepare_base_model
        # Creates prepare_base_model folder inside artifacts folder.
        create_directories([config.root_dir])

        # Data encapsulation into PrepareBaseModelConfig Class.
        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES,
            params_acceptable_classes=self.params.ACCEPTABLE_CLASSES,
            params_pooling=self.params.POOLING
        )

        return prepare_base_model_config
    
    def get_prepare_callback_config(self) -> PrepareCallbacksConfig:
        """
        Encapsulate callback configurations from config.yaml file.
        Returns:
            PrepareCallbacksConfig Object
        """
        # prepare_callbacks config.
        config = self.config.prepare_callbacks
        model_ckpt_dir = os.path.dirname(config.checkpoint_model_filepath)
        create_directories([
            Path(model_ckpt_dir),
            Path(config.tensorboard_root_log_dir)
        ])

        prepare_callback_config = PrepareCallbacksConfig(
            root_dir=Path(config.root_dir),
            tensorboard_root_log_dir=Path(config.tensorboard_root_log_dir),
            checkpoint_model_filepath=Path(config.checkpoint_model_filepath)
        )

        return prepare_callback_config
