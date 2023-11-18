"""This file contains ConfigurationManager Class"""
from cnnClassifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import DataIngestionConfig

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
    