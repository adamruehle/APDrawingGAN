import mtcnn
import os
import numpy as np
import cv2
import scipy.io

def detect_facial_landmarks(face_photo_file):
    detector = mtcnn.MTCNN()
    image = cv2.imread(face_photo_file)
    faces = detector.detect_faces(image)
    for face in faces:
        landmarks = face['keypoints']
        # Convert landmarks to a list of coordinate pairs
        landmark_values = []
        for keypoint in landmarks:
            x, y = landmarks[keypoint]
            landmark_values.append([x, y])
        print(landmark_values)
        # Save the extracted landmark values as a .mat file
        landmark_filename = face_photo_file.split(".")[0] + "_facial5point.mat"
        scipy.io.savemat(landmark_filename, {"landmarks": landmark_values})
        return landmark_filename

if __name__ == "__main__":
    face_photo_file = "myImages"
    detect_facial_landmarks(face_photo_file)
