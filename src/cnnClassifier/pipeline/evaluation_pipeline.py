"""This file contains EvaluationPipeline class"""
from cnnClassifier import logger
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.evaluation import Evaluation
STAGE_NAME = "Evaluation Stage"

class EvaluationPipeline:
    """
    This is the evaluation pipeline for the model. 
    """
    def __init__(self):
        pass
    def start(self):
        """
        Starts the evaluation process of the trained model on a test dataset.
        """
        logger.info("Starting %s", STAGE_NAME)
        config = ConfigurationManager()
        val_config = config.get_validation_config()
        evaluation = Evaluation(val_config)
        score = evaluation.evaluation()
        evaluation.save_score(score)
        logger.info("%s completed successfully.", STAGE_NAME)

if  __name__ == '__main__':
    try:
        evaluation_object = EvaluationPipeline()
        evaluation_object.start()
    except Exception as e:
        logger.exception(e)
        raise e