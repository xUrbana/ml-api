import requests
from PIL import Image
import numpy as np
from io import BytesIO
import tensorflow as tf

model = tf.keras.models.load_model('mnist.model')

r = requests.post('http://127.0.0.1:5000/MNIST/predict', files={'file': img})

if r.status_code == 200:
    print(r.json())
else:
    print(f'Error: {r.status_code}')
