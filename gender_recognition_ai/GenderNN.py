import sklearn
import tensorflow as tf
from tensorflow import keras
import os
import PIL
import face_recognition as fr
import numpy as np


# Splits up the data into training data and testing data with an 80/20 ratio then passes the data
# into a 3 layer sequential neural network with 128 input neurons, 1 hidden layer with 65 neurons,
# and a single output neuron that outputs a value between 0 and 1.
def train_model(faces, labels):
    train_images, test_images, train_labels, \
        test_labels = sklearn.model_selection.train_test_split(faces, labels, test_size=0.2)
    layer1 = keras.layers.Dense(128)
    layer2 = keras.layers.Dense(65, activation="relu")
    layer3 = keras.layers.Dense(1, activation="sigmoid")

    model = keras.Sequential([layer1, layer2, layer3])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(train_images, train_labels, epochs=6)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print("Tested acc: ", test_acc * 100, "%")

    model.save("model.h5")


# Gets the 128 length encoding of a face in a given image and then attempts to make an accurate
# prediction based on the output of the final neuron where female = (0, 0.5] and male = (0.5, 1)
def make_prediction(image_path="girl.jpg"):
    if "model.h5" not in os.listdir():
        print("Please run start.py to train a model before attempting to load one!")
        exit("No model found")
    model = keras.models.load_model("model.h5")
    try:
        image = fr.load_image_file(image_path)
        face_encoding = fr.face_encodings(image)
        if len(face_encoding) > 0:
            print("Face was found at ", image_path)
            face_encoding = face_encoding[0]
        else:
            return "none"
    except PIL.UnidentifiedImageError as e:
        print(e)
        exit("Unidentified Image Format")

    face_encoding = np.array(face_encoding)
    predict = model.predict(face_encoding.reshape(1, 128))
    if predict[0] > 0.5:
        return "male"
    else:
        return "female"
