from flask import Flask, request, jsonify, flash, redirect
from flask_apispec import FlaskApiSpec, use_kwargs, doc
import apispec
from apispec.ext.marshmallow import MarshmallowPlugin
from apispec import APISpec
from marshmallow import fields
from werkzeug.datastructures import FileStorage
from ml_api.utils import read_digit
from PIL import Image, ImageOps
import tensorflow as tf
import numpy as np
import markdown
from pygments.formatters.html import HtmlFormatter

################## Load our machine learning models to use for predictions ###################
model = tf.keras.models.load_model('mnist.model')
pmodel = tf.keras.models.Sequential([
    model,
    tf.keras.layers.Softmax()
])
##############################################################################################

########### Initialize app and inject API spec and allow for automatic file upload ###########
app = Flask(__name__)
docs = FlaskApiSpec(app)

file_plugin = MarshmallowPlugin()

app.config.update({
    'APISPEC_SPEC': APISpec(
        title='MNIST Hand-Written Digits Machine Learning API',
        version='v1',
        openapi_version=apispec.__version__,
        plugins=(file_plugin,)        
    )
})


@file_plugin.map_to_openapi_type('file', None)
class FileField(fields.Raw):
    pass
##############################################################################################


#################### Custom styling for code blocks generated in markdown ####################
formatter = HtmlFormatter(style='emacs', full=True, cssclass='codehilite')
css = f'<style>{formatter.get_style_defs()}</style>'
##############################################################################################

@docs.register
@doc(description='Make a prediction against the model with your own hand-drawn digit. Native 28x28 pixel images work the best. Images that are not 28x28 might suffer from downscaling.',
     tags=['files'], consumes=['multipart/form-data'])
@app.route('/MNIST/predict', methods=['POST'])
@use_kwargs({'file': FileField(required=True), 'encoding': fields.Str()}, locations=['files'])
def mnist_predict(file: FileStorage, encoding: str = 'L'):

    digit_data = read_digit(file.stream.read(), encoding)

    prediction = pmodel(digit_data)
    arr = prediction.numpy()[0]
    highest = np.argmax(arr)

    return jsonify({
        'prediction': int(highest),
        'confidence': arr.tolist()
    })

@docs.register
@doc(description='Information about the machine learning model used to make predictions.')
@app.route('/MNIST', methods=['GET'])
def mnist_specs():
    cfg = model.get_config()
    return jsonify(cfg)

@docs.register
@app.route('/', methods=['GET'])
def mnist_about():

    with open('README.md') as fp:
        md = fp.read()
    
    md = markdown.markdown(md, extensions=['codehilite'])

    return css + md


if __name__ == '__main__':
    app.run(debug=True)