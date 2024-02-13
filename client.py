"""
Client interface for GAN model"""

import os
import logging
import io
import base64
import json
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template
from flask_cors import cross_origin
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
LATENT_DIM = 128
app = Flask(__name__)
load_dotenv()
URL = os.getenv("CLOUD_RUN_URL")


@app.route("/")
def home():
    """
    Home Endpoint"""
    return render_template("index.html")


def get_data():
    '''
    Fetch data from the cloud run service and return it as a numpy array.'''
    response = requests.get(URL, timeout=200)
    if response.status_code == 200:
        lines = response.text.splitlines()
        data_sample = [json.loads(line) for line in lines]
        return np.array(data_sample)
    else:
        print("Request failed with status code", response.status_code)
        return None

@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    """
    Prediction Endpoint"""
    logging.info("Request Received, predicting ...")
    predictions = get_data().reshape(28, 28)
    if predictions is not None:
        img_io = io.BytesIO()

        plt.imsave(img_io, predictions, cmap="gray", format="png")
        img_base64 = base64.b64encode(img_io.getvalue()).decode("utf-8")
        logging.info("Done")
        return jsonify({"image": img_base64})
    return jsonify({"error": "Failed to get predictions"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
