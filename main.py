"""This file is the main file which intiates all the pipelines."""
from src.cnnClassifier import logger
from src.cnnClassifier.pipeline.data_ingestion_pipeline import DataIngestionPipeline

try:
    data_ingestion_object = DataIngestionPipeline()
    data_ingestion_object.start()
except Exception as e:
    logger.exception(e)
    raise e
