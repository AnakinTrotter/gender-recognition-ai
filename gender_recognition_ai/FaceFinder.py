import face_recognition as fr
import os
import PIL
import scipy.io.matlab as sp
import numpy as np

pics_folder = os.path.join("..", "data", "wiki")
labels_file = os.path.join(pics_folder, "wiki.mat")

image_paths = []


# Navigates to the pictures folder then returns a list of all the face pixel data
# The number of faces returned is not more than max_photos in most conditions
# A value of -1, the default value, for max_photos gets faces from all the photos
def get_face_encodings(max_photos=-1, start_folder=0, end_folder=100):
    faces = []
    # Iterates the folders start_folder to end_folder in the wiki directory
    dirs = os.listdir(pics_folder)
    print("Opening data folder: ", dirs)
    for i in range(start_folder, end_folder):
        directory = dirs[i]
        print("Loading files from: ", directory)
        numbered = os.path.join(pics_folder, directory)
        num_dirs = os.listdir(numbered)
        # Iterates through all the jpg files in each numbered directory
        for a in num_dirs:
            image_path = os.path.join(numbered, a)
            local_path = image_path.replace("\\", "/").replace("../data/wiki/", "")
            # Tries to load an image from the image_path
            try:
                image = fr.load_image_file(image_path)
                face_encoding = fr.face_encodings(image)
                if len(face_encoding) > 0:
                    print("Face was found at ", image_path)
                    face_encoding = face_encoding[0]
                    faces.append(face_encoding)
                    image_paths.append(local_path)
                if -1 < max_photos <= len(faces):
                    return faces
            except PIL.UnidentifiedImageError as e:
                # If the image_path turns out to be invalid, closes the program
                # If the image_path points to a .mat or .md file, just skip over it
                if "mat" and "md" not in str(image_path).split("."):
                    print(e)
                    exit("Unidentified Image Format")
    return faces


# Navigates to the data folder and finds the .mat file associated with the pictures
# then parses out the gender labels and returns them as a list
def get_face_labels():
    if len(image_paths) < 1:
        print("Please get face encodings before getting face labels.")
        exit("No face data detected!")
    gender_labels = []
    labels = list(sp.loadmat(labels_file)["wiki"]["gender"][0][0][0])
    paths = list(sp.loadmat(labels_file)["wiki"]["full_path"][0][0][0])
    for a in image_paths:
        gender_labels.append(labels[paths.index(a)])
    return gender_labels
