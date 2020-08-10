import pandas as pd
import numpy as np


# Converts the faces and labels lists into a pandas dataset
# which is then written to the disk as either a json or a csv
# or returned as a String if no valid format is given
def write_to_file(faces, labels, out_format="csv"):
    dataset = pd.DataFrame({"faces": faces, "labels": labels}, columns=["faces", "labels"])
    if out_format == "csv":
        dataset.to_csv("data.csv", index=False)
    elif out_format == "json":
        dataset.to_json("data.json")
    else:
        return dataset


# Reads the data from the csv file and converts it back into a list
def read_from_file(file_path="data.csv"):
    # Decided that csv is better than json so no json support for now
    data = pd.read_csv(file_path)
    data = data[["faces", "labels"]]
    return data


# Takes in list data such as from the read_from_file method and then
# formats the data into a list of list of length 128 floats
def format_face_data(data):
    faces = []
    face_data = np.array(data.drop(["labels"], 1))
    for a in face_data:
        arr = a[0].split()
        arr[0] = arr[0].replace("[", "")

        # remove all the random characters
        if "]" in arr:
            arr.remove("]")
        if "[" in arr:
            arr.remove("[")
        if "" in arr:
            arr.remove("")
        if " " in arr:
            arr.remove(" ")

        # Convert the Strings to floats and try to handle
        # any random characters that still made it in
        for i in range(0, len(arr)):
            if "]" in arr[i]:
                arr[i] = arr[i].replace("]", "")
            if "[" in arr[i]:
                arr[i] = arr[i].replace("[", "")
            if "\"" in arr[i]:
                arr[i] = arr[i].replace("\"", "")
        for i in range(0, len(arr)):
            try:
                arr[i] = float(arr[i])
            except ValueError as e:
                arr[i] = float(-69.6969)
                print(arr)
        # Cut the array down to 128 on the off chance
        # that some extra stragglers made it in
        if len(arr) != 128:
            print(arr)
        del arr[128:]
        faces.append(arr)
    return faces


# Get the label list and covert them all to floats
def format_label_data(data):
    labels = []
    label_data = np.array(data.drop(["faces"], 1))
    for a in label_data:
        labels.append(float(a))
    return labels


def clean_data(faces, labels):
    clean_faces = []
    clean_labels = []
    for i in range(0, len(labels)):
        if not np.isnan(labels[i]) and float(-69.6969) not in faces[i]:
            clean_faces.append(faces[i])
            clean_labels.append(labels[i])
    return [clean_faces, clean_labels]
