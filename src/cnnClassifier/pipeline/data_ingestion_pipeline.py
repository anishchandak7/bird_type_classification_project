"""This file contains DataIngestionPipeline Class"""
from cnnClassifier import logger
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_ingestion import DataIngestion

STAGE_NAME = 'Data Ingestion Stage'

class DataIngestionPipeline:
    """Data Ingestion functionalities."""
    def __init__(self):
        pass
    def start(self):
        """This function contains the code to initiate the data ingestion process."""
        logger.info("Starting %s", STAGE_NAME)
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(ingestion_config=data_ingestion_config)
        data_ingestion.download_data()
        logger.info("%s is completed.", STAGE_NAME)

if __name__ == '__main__':
    try:
        data_ingestion_object = DataIngestionPipeline()
        data_ingestion_object.start()
    except Exception as e:
        logger.exception(e)
        raise e
    