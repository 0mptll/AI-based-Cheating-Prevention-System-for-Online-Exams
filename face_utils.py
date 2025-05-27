import dlib
import numpy as np
import cv2
import os

predictor_path = "models/shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "models/dlib_face_recognition_resnet_model_v1.dat"

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(predictor_path)
face_encoder = dlib.face_recognition_model_v1(face_rec_model_path)


def get_face_descriptor(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector(rgb)

    if len(faces) != 1:
        return None

    shape = shape_predictor(rgb, faces[0])
    descriptor = face_encoder.compute_face_descriptor(rgb, shape)
    return np.array(descriptor)


def save_encoding(name, descriptor):
    os.makedirs("encodings", exist_ok=True)
    path = os.path.join("encodings", f"{name}.npy")
    np.save(path, descriptor)


def load_encoding(name):
    path = os.path.join("encodings", f"{name}.npy")
    if not os.path.exists(path):
        return None
    return np.load(path)


def compare_encodings(known, unknown, threshold=0.5):
    distance = np.linalg.norm(known - unknown)
    print(f"[DEBUG] Face encoding distance: {distance:.4f}")
    return distance < threshold
