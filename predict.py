import requests
from PIL import Image
import numpy as np
from io import BytesIO



img = Image.open('example-digits/4.png').convert('L').resize((28, 28))


data = {
    'encoding': 'L',
}

files = {
    'file': BytesIO(img.tobytes())
}

r = requests.post('http://127.0.0.1:5000/MNIST/predict', data=data, files=files)

if r.status_code == 200:
    print(r.json())
else:
    print(f'Error: {r.status_code}')