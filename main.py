"""
Code for web application and prediction.
"""

import io
import logging
import numpy as np
from flask import Flask, send_file, render_template
from tensorflow import keras

from PIL import Image

app = Flask(__name__)
model = keras.models.load_model("models/mnist_generator.h5", compile=False)
LATENT_DIM = 128
np.random.seed(42)
logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(message)s",
    level=logging.DEBUG,
    datefmt="%m/%d/%Y %I:%M:%S ",
)


@app.route("/", methods=["GET"])
def read_root():
    """
    Home page
    """
    return render_template("predict.html")


def sample_noise(n_samples: int, latent_dim: int):
    """
    Samples noise in the shape of (n_samples, latent_dim).
    """
    return np.random.rand(n_samples * latent_dim)


@app.route("/predict/", methods=["POST"])
def predict():
    """
    Returns a generated image from the model.

    Samples noise from a normal distribution and passes it through the
    generator model to get a generated image.
    """
    input_noise = sample_noise(1, LATENT_DIM).reshape((1, LATENT_DIM))
    prediction = model.predict(input_noise) * 255
    logging.info("Got predictions array")
    prediction = prediction.reshape(28, 28)
    predicted_image = Image.fromarray(prediction.astype(np.uint8))
    predicted_image = predicted_image.convert("L")

    img_buffer = io.BytesIO()
    predicted_image.save(img_buffer, format="JPEG")
    logging.info("Image saved to buffer")
    img_buffer.seek(0)

    # Return the binary image data directly
    return send_file(img_buffer, mimetype="image/jpeg")


# if __name__ == "__main__":
#     app.run(debug=False)
