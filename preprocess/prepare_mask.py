import cv2
import mtcnn
import numpy as np
import os

def detect_face(image_path):
  image = cv2.imread(image_path)
  detector = mtcnn.MTCNN()
  faces = detector.detect_faces(image)
  if faces:
    x, y, w, h = faces[0]['box']
    return x, y, w, h
  else:
    return None
  
def create_background_mask(image_path):
  face_coordinates = detect_face(image_path)

  if face_coordinates is not None:
    image = cv2.imread(image_path)
    mask = np.zeros_like(image)
    x, y, w, h = face_coordinates
    points = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])

    # Calculate Bezier control points
    control_points = [
      (points[0] + points[1]) // 2,
      (points[1] + points[2]) // 2,
      (points[2] + points[3]) // 2,
      (points[3] + points[0]) // 2
    ]

    # Draw Bezier curves
    cv2.polylines(mask, [points.astype(int)], isClosed=True, color=(255, 255, 255), thickness=2)
    cv2.polylines(mask, [np.array(control_points).astype(int)], isClosed=True, color=(255, 255, 255), thickness=2)
    cv2.fillPoly(mask, [np.array(control_points).astype(int)], (255, 255, 255))
    return mask
  else:
    return None
