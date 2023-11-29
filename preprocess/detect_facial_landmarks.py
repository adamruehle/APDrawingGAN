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

        is_image = face_photo_filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))

        if is_image:
            image = cv2.imread(face_photo_path)

            faces = detector.detect_faces(image)

            for face in faces:
                landmarks = face['keypoints']

                # Extract landmark values
                landmark_values = []
                print(landmarks)
                for keypoint in landmarks:
                    
                    x, y = landmarks[keypoint]
                    landmark_values.append([x, y])

                # Save the extracted landmark values as a .mat file
                facial_landmarks_filename = face_photo_filename.split(".")[0] + "_facial5point.mat"
                facial_landmarks_path = os.path.join(face_photo_dir, facial_landmarks_filename)
                print(landmark_values)
                scipy.io.savemat(facial_landmarks_path, {"landmarks": landmark_values})

if __name__ == "__main__":
    face_photo_dir = "myImages"
    detect_facial_landmarks(face_photo_dir)
