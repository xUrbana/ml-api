import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(
    optimizer='adam', 
    loss=loss_fn, 
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=10)
model.summary()
model.save('mnist.model')

'''
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

probability_model.save('mnist_probability.model')
'''


