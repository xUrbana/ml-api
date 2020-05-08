from flask import Flask, request, jsonify, flash, redirect
from flask_apispec import FlaskApiSpec
import base64
from PIL import Image, ImageOps
import tensorflow as tf
import numpy as np
import markdown

# Load models
model = tf.keras.models.load_model('mnist.model')

pmodel = tf.keras.models.Sequential([
    model,
    tf.keras.layers.Softmax()
])

# Initialize app and api
app = Flask(__name__)
docs = FlaskApiSpec(app)

def read_digit(data: bytes, encoding: str) -> np.array:
    im = Image.frombytes(encoding, (28, 28), data)
    im = ImageOps.invert(im)
    arr = np.array(im)
    arr = arr / 255.0
    arr[arr != 0.0] = 1.0
    print(arr)
    return np.array([arr])

@app.route('/MNIST/predict', methods=['POST'])
def mnist_predict():
    
    try:
        file = request.files['file']
    except KeyError:
        flash('No file part')
        return redirect(request.url)

    try:
        encoding = request.form['encoding']
    except KeyError:
        encoding = 'L'

    digit_data = read_digit(file.stream.read(), encoding)

    prediction = pmodel(digit_data)
    arr = prediction.numpy()[0]
    highest = np.argmax(arr)

    return jsonify({
        'prediction': int(highest),
        'confidence': arr.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)