import mtcnn
import os
import numpy as np
import cv2
import scipy.io

def detect_facial_landmarks(face_photo_dir):
    detector = mtcnn.MTCNN()

    face_photos = os.listdir(face_photo_dir)

    for face_photo_filename in face_photos:
        face_photo_path = os.path.join(face_photo_dir, face_photo_filename)
        image = cv2.imread(face_photo_path)

        face = detector.detect_faces(image)
        landmarks = face['keypoints']

        # Save the detected facial landmarks as a .mat file
        facial_landmarks_filename = face_photo_filename.split(".")[0] + "_facial5point.mat"
        facial_landmarks_path = os.path.join(face_photo_dir, facial_landmarks_filename)
        scipy.io.savemat(facial_landmarks_path, {"landmarks": landmarks})

if __name__ == "__main__":
    face_photo_dir = "myImages"
    detect_facial_landmarks(face_photo_dir)
