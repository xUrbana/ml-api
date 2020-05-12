import requests
from PIL import Image
import numpy as np
from io import BytesIO
import tensorflow as tf

model = tf.keras.models.load_model('mnist.model')

def read_digit(data: bytes, encoding: str) -> np.array:
    im = Image.frombytes(encoding, (28, 28), data)
    im = ImageOps.invert(im)
    arr = np.array(im)
    arr = arr / 255.0
    arr[arr != 0.0] = 1.0
    return np.array([arr])


img = read_digit('1.png')

r = requests.post('http://127.0.0.1:5000/MNIST/predict', files={'file': img})

if r.status_code == 200:
    print(r.json())
else:
    print(f'Error: {r.status_code}')
