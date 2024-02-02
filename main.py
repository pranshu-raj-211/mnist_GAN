import io
import logging
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image


app = FastAPI()
templates = Jinja2Templates(directory="templates")

model = load_model("mnist_generator.h5")
LATENT_DIM = 128
np.random.seed(42)
logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(message)s",
    level=logging.DEBUG,
    datefmt="%m/%d/%Y %I:%M:%S ",
)


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Home page"""
    return templates.TemplateResponse("predict.html", {"request": request})


def sample_noise(n_samples: int, latent_dim: int):
    """
    Samples noise in the shape of (n_samples, latent_dim)."""
    return np.random.rand(n_samples * latent_dim)


@app.post("/predict/")
async def predict():
    """
    Returns a generated image from the model.

    Samples noise from a normal distribution and passes it through the
    generator model to get a generated image."""
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
    return StreamingResponse(img_buffer, media_type="image/jpeg")
