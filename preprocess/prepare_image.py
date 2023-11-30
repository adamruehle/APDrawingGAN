# import matlab.engine
from . import detect_facial_landmarks as dfl
import argparse
import os
import numpy as np
import cv2
from . import prepare_mask
from . import face_align_512
import scipy.io

def prepare_image(path):
  file = dfl.detect_facial_landmarks(path)
  landmarks = scipy.io.loadmat(file)
  # print(landmarks)
  aligned_image_path = face_align_512.face_align_512(path, landmarks, "dataset/data/test_single", "dataset/landmark/ALL")
  print(aligned_image_path)
  background_mask = prepare_mask.segment_person(aligned_image_path)
  # save background mask to /dataset/mask/ALL
  name = os.path.splitext(os.path.basename(path))[0]
  cv2.imwrite("dataset/mask/ALL/" + name + "_aligned.png", background_mask)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('path', type=str)
  args = parser.parse_args()
  path = args.path
  print(path)
  prepare_image(path)