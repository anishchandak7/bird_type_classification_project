"""This file is the main file which intiates all the pipelines."""
from src.cnnClassifier import logger
from src.cnnClassifier.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from src.cnnClassifier.pipeline.prepare_base_model_pipeline import PrepareBaseModelPipeline
from src.cnnClassifier.pipeline.training_pipeline import TrainingPipeline
from src.cnnClassifier.pipeline.evaluation_pipeline import EvaluationPipeline

try:
    data_ingestion_object = DataIngestionPipeline()
    data_ingestion_object.start()
except Exception as e:
    logger.exception(e)
    raise e

try:
    base_model_prep_object = PrepareBaseModelPipeline()
    base_model_prep_object.initiate()
except Exception as e:
    logger.exception(e)
    raise e

try:
    training_object = TrainingPipeline()
    training_object.begin()
except Exception as e:
    logger.exception(e)
    raise e

try:
    evaluation_object = EvaluationPipeline()
    evaluation_object.start()
except Exception as e:
    logger.exception(e)
    raise e