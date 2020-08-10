from gender_recognition_ai import FaceFinder, DataHandler
import sklearn
from sklearn import neural_network
from sklearn.utils import shuffle

training_cases = 6
start = 20
end = 21

# labels = FaceFinder.get_face_labels(training_cases, start, end)
# print(labels)
# print(len(labels))

faces = FaceFinder.get_face_encodings(training_cases, start, end)

print("Data has been GOTTEN!")
print("Writing data...")
DataHandler.write_to_file(faces, labels)

print("Testing data reading...")
data = DataHandler.read_from_file()

faces = DataHandler.format_face_data(data)
labels = DataHandler.format_label_data(data)

train_images, test_images, train_labels, test_labels = sklearn.model_selection.train_test_split(faces, labels, test_size=0.1)

print(train_images)
print(train_labels)
print(test_images)
print(test_labels)
print("\nSuccess!\n")
