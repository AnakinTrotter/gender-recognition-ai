from gender_recognition_ai import FaceFinder

# test
faces = FaceFinder.get_face_encodings(1)
for a in faces:
    print(a)
