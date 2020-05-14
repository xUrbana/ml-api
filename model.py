import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf

print(f'Running TensorFlow version {tf.__version__}')

import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

shape = (28, 28, 1)

x_train = x_train.reshape(x_train.shape[0], *shape)
x_test = x_test.reshape(x_test.shape[0], *shape)

x_train //= 255
x_test //= 255

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=shape),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001), 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

model.fit(x=x_train, y=y_train, epochs=10)

model.evaluate(x_test, y_test)

model.save('model')



