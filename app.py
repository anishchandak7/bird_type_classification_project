"""This file contains code for API handling using flask."""
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decode_image
from cnnClassifier.pipeline.prediction_pipeline import PredictionPipeline

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

class ClientApp:
    """
    This class is responsible for handling the client requests.
    """
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)

@app.route('/', methods=['GET'])
@cross_origin()
def home():
    """
    This function returns the home page of the app.
    """
    return render_template('index.html')

@app.route("/train", methods=['GET','POST'])
@cross_origin()
def train_route():
    """
    This function handles training route.
    """
    os.system("python main.py")
    return "Training completed Successfully!"

@app.route("/predict", methods=['POST'])
@cross_origin()
def prediction_route():
    """
    This function handles prediction route.
    """
    if "artifacts" not in os.listdir(os.getcwd()):
        os.system("python main.py")
        return render_template('index.html', retrained_message="Artifacts were missing, Hence model was retrained!")
    image = request.json['image']
    decode_image(image, clApp.filename)
    result = clApp.classifier.predict()
    return jsonify(result)


if __name__ == '__main__':
    clApp = ClientApp()
    app.run(host="0.0.0.0", port=8080)
