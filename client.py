"""
Client interface for GAN model"""

import logging
import base64
import io
import json
from flask import Flask, jsonify, render_template
from flask_cors import CORS, cross_origin
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
    logging.warning("At home")
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    """
    Prediction Endpoint"""
    logging.warning("Request received")
    input_data = np.random.rand(1, LATENT_DIM).tolist()

    # Convert the data to the format expected by TensorFlow Serving
    data = {
        "signature_name": "serving_default",
        "instances": input_data,
    }

    # Convert the data to JSON
    data = json.dumps(data)

    # Send a POST request to the TensorFlow Serving model server
    headers = {"content-type": "application/json"}
    json_response = requests.post(
        "http://localhost:8501/v1/models/my_model:predict",
        data=data,
        headers=headers,
        timeout=60,
    )

    # Parse the response
    predictions = json.loads(json_response.text)["predictions"]

    # Convert the predictions to a numpy array
    predictions = np.array(predictions)

    # Reshape the predictions if necessary
    predictions = predictions.reshape((28, 28))

    # Create a BytesIO object
    img_io = io.BytesIO()

    # Save the image to the BytesIO object
    plt.imsave(img_io, predictions, cmap="gray", format="png")
    # plt.imshow(predictions)
    # plt.show()

    # Get the base64 encoding of the image
    img_base64 = base64.b64encode(img_io.getvalue()).decode("utf-8")

    # Return the base64 image in the response
    return jsonify({"image": img_base64})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
