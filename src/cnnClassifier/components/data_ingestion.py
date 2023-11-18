"""This file contains data ingestion class."""
import os
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import DataIngestionConfig
os.environ['KAGGLE_CONFIG_DIR'] = os.path.join(os.getcwd(), '.kaggle')
import kaggle

class DataIngestion:
    """Data Ingestion functionalities"""
    def __init__(self, ingestion_config: DataIngestionConfig):
        self.ingestion_config = ingestion_config
    def download_data(self):
        """Downloads the data from Kaggle"""
        # Authentication using kaggle.json
        kaggle.api.authenticate()
        logger.info("Kaggle Authentication successful.")

        # Download the dataset.
        logger.info("Downloading data from kaggle...")
        kaggle.api.dataset_download_files(self.ingestion_config.source_url,
                                          self.ingestion_config.unzip_dir,
                                          unzip=True)
        logger.info('%s dataset downloaded and saved at %s',
                    self.ingestion_config.source_url, self.ingestion_config.root_dir)
    