from gender_recognition_ai import FaceFinder, DataHandler
import sklearn
import os
import tensorflow as tf
from tensorflow import keras
from sklearn import neural_network
from sklearn.utils import shuffle

overwrite = False

if "data.csv" in os.listdir():
    print("\nA data.csv file was detected.")
    print("Would you like to overwrite it? (y/n)")
    should_prompt = True
    while should_prompt:
        response = input()
        if response == "y":
            overwrite = True
            should_prompt = False
        elif response == "n":
            overwrite = False
            should_prompt = False
        else:
            print("Please respond with 'y' for yes or 'n' for no.")

if overwrite:
    print("\n")
    print("Please enter which folder to start on/end on.")
    print("The starting folder is inclusive while the ending folder is not.")
    print("ex: [start, end)")
    print("It's like substring in Java.")
    print("\n")

    should_prompt = True
    start = -1
    end = -1

    while should_prompt:
        try:
            start = int(input("Starting folder: "))
            end = int(input("Ending folder: "))
        except ValueError:
            print("Input must be an int between 0 and 100!\n")
            continue
        if -1 < start < end <= 100:
            should_prompt = False
        else:
            print("Invalid input!\n")

    print("\nProcess started... this may take several hours.\n")
    faces = FaceFinder.get_face_encodings(start_folder=start, end_folder=end)
    labels = FaceFinder.get_face_labels()

    print("Corresponding genders: ", labels)

    print("\nData has been GOTTEN!")
    print("\nWriting data...")
    DataHandler.write_to_file(faces, labels)
    print("Data saved to data.csv")

print("\nReading data...\n")
data = DataHandler.read_from_file()

faces = DataHandler.format_face_data(data)
labels = DataHandler.format_label_data(data)

clean_data = DataHandler.clean_data(faces, labels)
faces = clean_data[0]
labels = clean_data[1]

print("\n")
print("Final face count: ", len(faces))
print("Final label count: ", len(labels))

train_images, test_images, train_labels, \
    test_labels = sklearn.model_selection.train_test_split(faces, labels, test_size=0.2)

print("\nSuccess!\n")

retrain = True

if "model.h5" in os.listdir():
    print("\nA model.h5 file was detected.")
    print("Would you like to retrain the model? (y/n)")
    should_prompt = True
    while should_prompt:
        response = input()
        if response == "y":
            retrain = True
            should_prompt = False
        elif response == "n":
            retrain = False
            should_prompt = False
        else:
            print("Please respond with 'y' for yes or 'n' for no.")

if retrain:
    layer1 = keras.layers.Dense(128)
    layer2 = keras.layers.Dense(65, activation="relu")
    layer3 = keras.layers.Dense(1, activation="sigmoid")

    model = keras.Sequential([layer1, layer2, layer3])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(train_images, train_labels, epochs=6)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print("Tested acc: ", test_acc*100, "%")

    model.save("model.h5")

print("\nDone!\n")
