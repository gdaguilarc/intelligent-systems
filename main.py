# Proyecto Final
# Clustering with kmeans a group of numbers from an image, crop and then predict the label with a NN

# Location:
# Tecnologico de Monterrey Campus Ciudad de México

# Contributors:
# Andrea Beatriz Becerra Bolaños - A01337434
# Guillermo David Aguilar Castilleja - A01337242

# Dependencies:
# tensorflow == 2.3.0

from datetime import date
import random
from pathlib import Path
from PIL import Image, ImageOps
from sklearn.cluster import KMeans
import numpy as np

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model

from matplotlib import pyplot as plt

# IMAGES
IMAGE_ONE = Image.open("img/digits-handmade.jpg")
IMAGE_TWO = Image.open("img/prueba1.png")
IMAGE_THREE = Image.open("img/prueba2.png")
IMAGE_FOUR = Image.open("img/prueba3.jpeg")


# NET CONSTANTS
NUMBER_LAYERS = 3
EPOCHS = 15
OPTIMIZER = "Adam"
DATE = date.today().strftime("%d-%m-%Y")
FILENAME = "DigitNN-Layers{}-{}-{}epochs-{}.h5".format(
    NUMBER_LAYERS, OPTIMIZER, EPOCHS, DATE)

BEST_NET = "DigitNN-3-adam-20-29-07-2020.h5"


# DATA
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# NORMALIZING DATA
x_train = x_train / 255
x_test = x_test / 255


# MODEL
def train(x_train, y_train, epochs=20, optimizer="adam", net=FILENAME):

    if net != FILENAME:
        my_file = Path("models/" + net)
    else:
        my_file = Path("models/" + FILENAME)

    if my_file.is_file():
        if net != FILENAME:
            model = load_model("models/" + net)
        else:
            model = load_model("models/" + FILENAME)

        return model

    model = Sequential()
    model.add(Flatten())
    model.add(Dense(392, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation="softmax"))
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy', metrics=["accuracy"])
    history = model.fit(x_train, y_train, epochs=epochs,
                        shuffle=True, validation_split=0.25)
    model.summary()
    model.save("models/"+FILENAME)
    print_history(history)
    model.evaluate(x_test, y_test)
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


def image_preprocessing(img, k, inverted=True):
    img = ImageOps.grayscale(img)
    if inverted:
        img = ImageOps.invert(img)
    img = np.asarray(img)
    pairs = pair_points(img)

    cluster_labels = cluster(pairs, k)

    images = []
    for i in range(k):
        images.append(cutted_digit(cluster_labels, img, pairs, i))

    return images


def cutted_digit(cluster, img, pairs, index, inverted=True):
    positions = np.where(cluster == index)
    img_boundaries = pairs[positions][:]

    # Get Square
    y_max = img_boundaries[:, 0].max()
    x_max = img_boundaries[:, 1].max()
    y_min = img_boundaries[:, 0].min()
    x_min = img_boundaries[:, 1].min()
    area = (x_min, y_min, x_max, y_max)

    cutted = Image.fromarray(img)
    cutted = cutted.crop(area)

    resized = cutted.resize((20, 20))
    # resized.show()  # Borrar

    resized = np.array(resized)

    resized = Image.fromarray(resized)
    resized = ImageOps.invert(resized)
    resized = np.asarray(resized)
    return Image.fromarray(np.pad(resized, ((4, 4), (4, 4)), "constant", constant_values=0))


def pair_points(data):
    points = []
    max_x = len(data)
    max_y = len(data[0])
    for i in range(max_x):
        for j in range(max_y):
            if data[i][j] < 125:
                points.append((i, j))

    return np.array(points)


def cluster(pairs, k):
    dbscan = KMeans(n_clusters=k)
    cluster = dbscan.fit_predict(pairs)
    plt.scatter(pairs[:, 1], pairs[:, 0], c=cluster, cmap='plasma')
    plt.show()
    return cluster


def predict_images(images):

    _, axs = plt.subplots(1, len(images))

    for i in range(len(images)):
        image = np.asarray(images[i])
        pred = model.predict(image.reshape(1, 28, 28, 1))
        axs[i].set_title(str(pred.argmax()))
        axs[i].imshow(image, cmap="gray")
        axs[i].axis('off')

    plt.show()





# TRAINING OR IMPORTING THE MODEL
model = train(x_train, y_train, EPOCHS, OPTIMIZER, net=FILENAME)

# BEST TRY
# DigitNN-3-adam-20-29-07-2020
# IMAGE_ONE 12/16 16 False 75%
# IMAGE_TWO 7/10 10 True  70% 
# IMAGE_THREE 8/10 10 True  80%

# BEST TRY
# DigitNN-Layers3-Adam-20epochs-30-07-2020
# IMAGE_ONE 12/16 16 False 75%
# IMAGE_TWO 7/10 10 True  70% 
# IMAGE_THREE 5/10 10 True  50%

# BEST TRY
# DigitNN-Layers3-Adam-15epochs-30-07-2020
# IMAGE_ONE 12/16 16 False 62%
# IMAGE_TWO 7/10 10 True 70% 
# IMAGE_THREE 7/10 10 True 70%

# # SECOND BEST
# DigitNN-3-Nadam-20epochs-30-07-2020
# IMAGE_ONE 12/16 16 False 75%
# IMAGE_TWO 8/10 10 True  80% 
# IMAGE_THREE 6/10 10 True  60%  


# THIRD BEST
# DigitNN-3-Adam-15epochs-30-07-2020
# IMAGE_ONE 11/16  16 False  68.75%
# IMAGE_TWO 7/10 10 True  70% 
# IMAGE_THREE 7/10 10 True  70% 
 
# IMAGE CLUSTERING AND SEGMENTATION
# CLUSTERING: KMEANS
images = image_preprocessing(IMAGE_ONE, 16, False)

# RESULTS
predict_images(images)
