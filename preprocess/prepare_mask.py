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
    cv2.rectangle(mask, (x, y), (y+w, y+h), (255, 255, 255), thickness=cv2.FILLED)
    return mask
  else:
    return None

if __name__ == "__main__":
  image_path = "myImages\\AdamRuehle\\data\\test_single\\AdamRuehle_aligned.png"
  background_mask = create_background_mask(image_path)

  if background_mask is not None:
    # Visualize the results
    cv2.imshow("Original Image", cv2.imread(image_path))
    cv2.imshow("Background Mask", background_mask)
    pathname = ".\\" + os.path.splitext(image_path)[0].split("\\")[1] + "\\" + os.path.splitext(image_path)[0].split("\\")[2] + "\\" + 'AdamR.png'
    print(pathname)
    # cv2.imwrite(pathname, background_mask)
  else:
    print("No face detected in the image.")