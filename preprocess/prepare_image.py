# import matlab.engine
import detect_facial_landmarks as dfl
import argparse
import os
import numpy as np
import cv2
import prepare_mask
import face_align_512
import scipy.io

def main(path):
  file = dfl.detect_facial_landmarks(path)
  landmarks = scipy.io.loadmat(file)
  # print(landmarks)
  aligned_image_path = face_align_512.face_align_512(path, landmarks, "dataset/data/test_single", "dataset/landmark/ALL")
  # print(aligned_image_path)
  background_mask = prepare_mask.create_background_mask(aligned_image_path)
  # save background mask to /dataset/mask/ALL
  name = os.path.splitext(os.path.basename(path))[0]
  cv2.imwrite("dataset/mask/ALL/" + name + "_mask.png", background_mask)

  # eng.quit()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('path', type=str)
  args = parser.parse_args()
  path = args.path
  print(path)
  main(path)