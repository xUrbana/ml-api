from flask import Flask, request, jsonify, flash, redirect
from flask_apispec import FlaskApiSpec
import base64
from PIL import Image
import tensorflow as tf
import numpy as np
import markdown

# Load model
model = tf.keras.models.load_model('mnist.model')

# Initialize app and api
app = Flask(__name__)
docs = FlaskApiSpec(app)

def read_digit(data: list, encoding: str) -> np.array:
    im = Image.frombytes(encoding, (28, 28), data)
    im.show()
    arr = np.array(im) / 255.
    return np.reshape(arr, (1, 28, 28))

@app.route('/MNIST/predict', methods=['POST'])
def mnist_predict():
    
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    encoding = request.form['encoding']
    digit_data = read_digit(file.stream.read(), encoding)
    prediction = model.predict(digit_data)
    print(prediction)
    highest = np.argmax(prediction)

    return jsonify({
        'prediction': int(highest),
        'confidence': prediction.tolist()[0]
    })

if __name__ == '__main__':
    app.run(debug=True)