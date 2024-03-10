import io
import logging
import numpy as np
from tensorflow import keras
from google.cloud import storage
import json


model= None
def sample_noise(n_samples: int, latent_dim: int):
    """
    Samples noise in the shape of (n_samples, latent_dim).
    """
    return np.random.rand(n_samples * latent_dim)

def main(request):
    """
    Main function for Google Cloud Functions.
    """
    LATENT_DIM = 128
    np.random.seed(42)
    global model
    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(message)s",
        level=logging.DEBUG,
        datefmt="%m/%d/%Y %I:%M:%S ",
    )

    # Create a storage client.
    storage_client = storage.Client()

    # TODO: replace with your bucket name and model file name
    bucket_name = "your-bucket-name"
    source_blob_name = "model-file-name"
    destination_file_name = "/tmp/model.h5"

    if model is None:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)

        model = keras.models.load_model(destination_file_name, compile=False)
    input_noise = sample_noise(1, LATENT_DIM).reshape((1, LATENT_DIM))
    prediction = model.predict(input_noise) * 255
    logging.info("Got predictions array")
    prediction = prediction.reshape(28, 28)

    # Convert the numpy array to a list, so it can be serialized to JSON
    prediction_list = prediction.tolist()

    # Return the prediction as a JSON response
    return json.dumps(prediction_list), 200, {'ContentType':'application/json'} 
