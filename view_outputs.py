'''
Testing code for cloud run deployment.'''
import json
import os
from dotenv import load_dotenv
import requests
import numpy as np
import matplotlib.pyplot as plt

load_dotenv()
URL = os.getenv("CLOUD_RUN_URL")


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


def plot_data(data):
    '''
    Plot the data as an image.'''
    image_data=data.reshape(28,28)
    plt.imshow(image_data, cmap="gray_r")
    plt.show()


if __name__=='__main__':
    data = get_data()
    plot_data(data)
