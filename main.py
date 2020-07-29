from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
from datetime import date

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model

from matplotlib import pyplot as plt


# DATA
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# NORMALIZING DATA
x_train = x_train / 255
x_test = x_test / 255


NUMBER_LAYERS = 3
EPOCHS = 15
OPTIMIZER = "adam"
DATE = date.today().strftime("%d-%m-%Y")
FILENAME = "DigitNN-{}-{}-{}-{}.h5".format(
    NUMBER_LAYERS, OPTIMIZER, EPOCHS, DATE)


# MODEL
def train(x_train, y_train, epochs=20, optimizer="adam"):
    my_file = Path(FILENAME)
    if my_file.is_file():
        model = load_model(FILENAME)
        return model

    model = Sequential()
    model.add(Flatten())
    model.add(Dense(392, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(196, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(20, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation="softmax"))
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy', metrics=["accuracy"])
    history = model.fit(x_train, y_train, epochs=epochs,
                        shuffle=True, validation_split=0.25)
    model.summary()
    model.save(FILENAME)
    print_history(history)
    return model


def print_history(history):
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.plot(
        range(len(history.history['accuracy'])), history.history['accuracy'])
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.subplot(2, 2, 2)
    plt.plot(range(len(history.history['loss'])), history.history['loss'])
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.show()


model = train(x_train, y_train, EPOCHS, OPTIMIZER)


model.evaluate(x_test, y_test)


pred = model.predict(x_test[88].reshape(1, 28, 28, 1))

plt.imshow(x_test[88], cmap="gray")
plt.title(str(pred.argmax()))
plt.show()
