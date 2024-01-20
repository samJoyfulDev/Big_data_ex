from tensorflow.keras.datasets import fashion_mnist
import imageio

(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()

for i in range(5):
    imageio.imwrite("uploads/{}.png ".format(i), im=X_test[i])

import os
import requests
import numpy as np
import tensorflow as tf

from imageio import imwrite, imread

from flask import Flask, request, jsonify

from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

x_train = x_train.reshape(-1,28*28)
x_train.shape
x_test = x_test.reshape(-1,28*28)
x_test.shape

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(units = 128, activation='relu', input_shape=(784,)))

model.add(tf.keras.layers.Dense(units = 64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(units = 64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(units = 64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.summary()

model.fit(x_train, y_train, epochs=10)

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print('Test accuracy: {}'.format(test_accuracy))

json_model = model.to_json()

with open('fashionmnist_model.json', 'w') as json_file:
    json_file.write(json_model)

model.save_weights('FashionMNIST_weights.h5')

with open('fashionmnist_model.json', 'r') as f:
    model_json = f.read()

model = tf.keras.models.model_from_json(model_json)

model.load_weights("FashionMNIST_weights.h5")



app = Flask(__name__)

@app.route("/api/v1/<string:img_name>", methods=["POST"])
def classify_image(img_name):
    upload_dir = "uploads/"
    image = imread(upload_dir + img_name)
    classes = ["T-shirt/top","Trouser","pullover","Dress","Coat","Sandal","Shirt","Bag", "Ankle boot"]

    prediccion = model.predict([image.reshape(1, 28*28)])
    return jsonify({"object_detected":classes[np.argmax(prediccion[0])]})

app.run(port=5000,debug=(False))