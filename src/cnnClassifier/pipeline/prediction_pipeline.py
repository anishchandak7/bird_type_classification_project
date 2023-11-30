"""This file contains code for prediction"""
import os
import numpy as np
import tensorflow as tf
from cnnClassifier.config.configuration import ConfigurationManager

class PredictionPipeline:
    """
    This class is used to create a prediction pipeline.
    """
    def __init__(self, file_name) -> None:
        self.file_name = file_name
        config = ConfigurationManager()
        predict_config = config.get_prediction_config()
        self.classes = predict_config.params_acceptable_classes
    def predict(self):
        """
        This method is used to make predictions on the given data set.
        """
        # load the model.
        model_path = os.path.join("artifacts", "training", "model.h5")
        model = tf.keras.models.load_model(model_path)

        # load test image.
        image_name = self.file_name
        test_image = tf.keras.preprocessing.image.load_img(image_name, target_size=(224,224))
        # Convert image to numpy array.
        image_array = tf.keras.preprocessing.image.img_to_array(test_image)
        image_array = np.expand_dims(image_array, axis=0)
        predictions = model.predict(image_array)
        print(predictions)
        result = np.argmax(predictions, axis=1)
        print("Result:",result)
        return [{"image":self.classes[result[0]]}]
