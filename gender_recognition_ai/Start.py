from gender_recognition_ai import FaceFinder
import matplotlib.pyplot as plt

# test
training_cases = 1

faces = FaceFinder.get_face_encodings(training_cases)
labels = FaceFinder.get_face_labels(training_cases)

