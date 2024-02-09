"""
Client interface for GAN model"""

import logging
import base64
import io
import json
from flask import Flask, jsonify, render_template
from flask_cors import cross_origin
import requests
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
LATENT_DIM = 128
app = Flask(__name__)


@app.route("/")
def home():
    """
    Home Endpoint"""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    """
    Prediction Endpoint"""
    logging.info("Request Received, predicting ...")
    input_data = np.random.rand(1, LATENT_DIM).tolist()

    data = {
        "signature_name": "serving_default",
        "instances": input_data,
    }
    data = json.dumps(data)

    headers = {"content-type": "application/json"}
    json_response = requests.post(
        "http://localhost:8501/v1/models/my_model:predict",
        data=data,
        headers=headers,
        timeout=60,
    )

    predictions = json.loads(json_response.text)["predictions"]
    predictions = np.array(predictions)
    predictions = predictions.reshape((28, 28))
    img_io = io.BytesIO()

    plt.imsave(img_io, predictions, cmap="gray", format="png")
    img_base64 = base64.b64encode(img_io.getvalue()).decode("utf-8")
    return jsonify({"image": img_base64})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
