"""This file contains evaluation class."""
from pathlib import Path
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from cnnClassifier.utils.common import generate_dataframe, save_json
from cnnClassifier.config.configuration import EvaluationConfig

class Evaluation:
    """
    This is the main class for evaluating a trained model on test data.
    """
    def __init__(self, config: EvaluationConfig):
        self.config = config
    def _valid_generator(self):
        """
        Generate test dataset using image augmentation techniques and batch size of 32.

        Returns:
           keras.src.preprocessing.image.DataFrameIterator: test_gen
        """
        test_df = generate_dataframe(
            self.config.test_data, self.config.params_acceptable_classes
        )

        generator_kwargs = dict(
            rescale = 1./255,
        )

        dataflow_kwargs = dict(
            x_col = 'filepath',
            y_col = 'label',
            target_size = self.config.params_image_size[:-1],
            color_mode = 'rgb',
            batch_size = self.config.params_batch_size
        )

        ts_gen = ImageDataGenerator(
            **generator_kwargs
        )

        test_gen = ts_gen.flow_from_dataframe(
            dataframe=test_df,
            shuffle=True,
            **dataflow_kwargs
        )

        return test_gen
    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        """
        Load saved model from disk.

        Args:
            path (Path): Model location.

        Returns:
            tf.keras.Model: loaded model
        """
        return tf.keras.models.load_model(path)
    def evaluation(self):
        """
        Evaluates the trained model on the test set.
        """
        model = self.load_model(self.config.path_of_model)
        test_gen = self._valid_generator()
        score = model.evaluate(test_gen)
        return score
    def save_score(self, score):
        """
        Save the accuracy of the model to a csv file.
        """
        scores = {"loss": score[0], "accuracy": score[1]}
        save_json(path=Path("scores.json"), data=scores)