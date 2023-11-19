"""This file contains Training class."""
from pathlib import Path
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from cnnClassifier.utils.common import generate_dataframe
from cnnClassifier.config.configuration import TrainingConfig

class Training:
    """
    This is the main training class for Classifier model.
    It uses TensorFlow and Keras to load and train the model.
    """
    def __init__(self, config: TrainingConfig):
        self.config = config
    def get_base_model(self):
        """
        Get base model from pre-trained weights.

        Returns:
            tf.keras.models.Model: loaded updated base model.
        """
        return tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
    def train_valid_generator(self):
        """
        Generate dataframes of training and validation sets using pandas dataframe.

        Returns:
            keras.src.preprocessing.image.DataFrameIterator: train_gen, valid_gen
        """
        train_df = generate_dataframe(
            self.config.training_data, self.config.params_acceptable_classes
        )

        valid_df = generate_dataframe(
            self.config.validation_data, self.config.params_acceptable_classes
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

        val_gen = ImageDataGenerator(**generator_kwargs)

        if self.config.params_is_augmentation:
            tr_gen = ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **generator_kwargs)
        else:
            tr_gen = ImageDataGenerator(**generator_kwargs)   
        valid_gen = val_gen.flow_from_dataframe(
            dataframe=valid_df,
            shuffle=False,
            **dataflow_kwargs
        )
        train_gen = tr_gen.flow_from_dataframe(
            dataframe=train_df,
            shuffle=True,
            **dataflow_kwargs
        )

        return (train_gen, valid_gen)
    def train(self, callback_list:list):
        """
        model training process.

        Args:
            callback_list (list)
        """

        train_gen, valid_gen = self.train_valid_generator()
        model = self.get_base_model()

        model.fit(
            x =  train_gen,
            verbose = 1,
            shuffle = False,
            epochs= self.config.params_epochs,
            validation_data = valid_gen,
            callbacks = callback_list,
            use_multiprocessing=True,
            workers=tf.data.experimental.AUTOTUNE
        )

        self.save_model(
            path = self.config.trained_model_path,
            model=model
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Save trained model to the specified file.

        Args:
            path (Path): Save path.
            model (tf.keras.Model): Model to save.
        """
        model.save(path)
