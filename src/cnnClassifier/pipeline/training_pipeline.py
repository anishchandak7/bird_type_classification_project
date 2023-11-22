"""This file contains TrainingPipeline class."""
from cnnClassifier import logger
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_callbacks import PrepareCallback
from cnnClassifier.components.training import Training

STAGE_NAME = 'Training Stage'

class TrainingPipeline:
    """
    This is the main pipeline for training a model using CNN Classifier.
    """
    def __init__(self) -> None:
        pass
    def begin(self):
        """
        Begin method of the TrainingPipeline. 
        It initializes all necessary components and starts the process.
        """
        logger.info("Starting %s", STAGE_NAME)
        config = ConfigurationManager()
        prepare_callbacks_config = config.get_prepare_callback_config()
        prepare_callbacks = PrepareCallback(config=prepare_callbacks_config)
        callback_list = prepare_callbacks.get_tb_ckpt_callbacks()
        logger.info('%s callbacks', str(len(callback_list)))
        training_config = config.get_training_config()
        training = Training(config=training_config)
        logger.info('Starting training process')
        training.train(
            callback_list=callback_list
        )
        logger.info("%s completed successfully.", STAGE_NAME)
    
if __name__ == '__main__':
    try:
        training_object = TrainingPipeline()
        training_object.begin()
    except Exception as e:
        logger.exception(e)
        raise e
