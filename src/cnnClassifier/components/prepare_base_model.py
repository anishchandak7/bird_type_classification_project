"""This file contains Base Model Creation Class."""
from pathlib import Path
import tensorflow as tf
from cnnClassifier.config.configuration import PrepareBaseModelConfig

class PrepareBaseModel:
    """
    This class contains the functionalities necessary to prepare a base model and full model.
    """
    def __init__(self, config: PrepareBaseModelConfig) -> None:
        self.config = config
        self.model = tf.keras.Model()
        self.full_model = tf.keras.Model()
    def get_base_model(self):
        """
        Function that returns a Efficientnet pre-trained model as base model.
        """
        self.model = tf.keras.applications.efficientnet.EfficientNetB0(
            include_top= self.config.params_include_top,
            weights= self.config.params_weights,
            input_shape= self.config.params_image_size,
            pooling= self.config.params_pooling)
        self.save_model(path=self.config.base_model_path, model=self.model)
    @staticmethod
    def _prepara_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for _ in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for _ in model.layers[:-freeze_till]:
                model.trainable = False

        full_model = tf.keras.models.Sequential([model,
            tf.keras.layers.BatchNormalization(axis= -1, momentum= 0.99, epsilon= 0.001),
            tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(l= 0.016),
                activity_regularizer=tf.keras.regularizers.l1(0.006),
                bias_regularizer=tf.keras.regularizers.l1(0.006), activation= 'relu'),
            tf.keras.layers.Dropout(rate= 0.45, seed= 123),
            tf.keras.layers.Dense(classes, activation= 'softmax')])
        # Compile the model.
        full_model.compile(tf.keras.optimizers.Adamax(learning_rate= learning_rate),
                           loss= 'categorical_crossentropy', metrics= ['accuracy'])
        # Generate model summary.
        full_model.summary()
        # Return complete model.
        return full_model
    def update_base_model(self):
        """
        Updates the base model with new layers according to the configuration file.
        """
        self.full_model = self._prepara_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Saves the model at the given path.

        Args:
            path (Path): Path to save the model.
            model (tf.keras.Model): Model
        """
        model.save(path)