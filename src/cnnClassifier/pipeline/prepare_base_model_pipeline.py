"""This file contains PrepareBaseModelPipeline class."""
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier import logger

STAGE_NAME = "Base Model Preparation Stage"

class PrepareBaseModelPipeline:
    """This class contains functionalities to prepare base model."""
    def __init__(self):
        pass
    def initiate(self):
        """Function initiatialize the base model creation."""
        logger.info("Starting %s", STAGE_NAME)
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        logger.info("Base model saved successfully.")
        prepare_base_model.update_base_model()
        logger.info("Updated model saved successfully.")
        logger.info("%s completed successfully.", STAGE_NAME)