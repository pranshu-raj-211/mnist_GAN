## Deployment

Deploying the GAN needs to be done in two parts - the model server and a client interface. The entire code of the client server as well as the training notebooks can be found [here](https://github.com/pranshu-raj-211/mnist_GAN).

For the deployment of the model server I will be using Google cloud's cloud functions. They work a lot like remote code executors and allow you to send requests to the functions inside.

## Cloud Function - Model Server

I have deployed the trained model on a gcp function. To do this we need to have a gcp account set up with a project created. Then the model needs to be uploaded to a bucket resource in the same project. I chose the h5 format as it allows me to call the model without having to reconstruct it again.

After this step I set up a cloud function service and allowed access by http requests for invocation to make it callable. The code to be executed loads the model, generates a random sample of noise and sends a numpy array of predictions back to the caller, in this case it is the client interface.

Here's a sample of the code that can be used to create a cloud function -

```python
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

    storage_client = storage.Client()

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

    prediction_list = prediction.tolist()

    return json.dumps(prediction_list), 200, {'ContentType':'application/json'}
```

## Building a client

The most basic version of such a client could be a python script with requests running which sends a noise sample to the model through a post request and receives the corresponding input.

But to build a presentable interface I'll use flask along with html templates to send requests to the model server and receive the corresponding generated image.

The interface is not very important right now, so I will just be adding a button which when clicked will call a predict api built in flask which calls the model server and receives the image, forwards it and displays on the interface.

The code for the flask client can be found [here](https://github.com/pranshu-raj-211/mnist_GAN/blob/main/client.py)and for the html template [here](https://github.com/pranshu-raj-211/mnist_GAN/blob/main/templates/index.html).
