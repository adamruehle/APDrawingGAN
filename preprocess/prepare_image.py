import matlab.engine
import detect_facial_landmarks as dfl
import argparse
import os
import numpy as np
import cv2
import prepare_mask

def main(path):
  file = dfl.detect_facial_landmarks(path)
  eng = matlab.engine.start_matlab()
  function_directory = 'preprocess'
  eng.addpath(os.path.abspath(function_directory), nargout=0)
  landmarks = eng.load(file)
  print(landmarks)
  # pathname = ".\\" + os.path.splitext(path)[0].split("\\")[1] + "\\" + os.path.splitext(path)[0].split("\\")[2]
  eng.face_align_512(path, landmarks, "dataset/data/test_single", "dataset/landmark/ALL")

  # aligned_image_path = pathname + "\\" + os.path.splitext(path)[0].split("\\")[2] + "_aligned.png"
  # print(aligned_image_path)
  # background_mask = prepare_mask.create_background_mask(aligned_image_path)
  # cv2.imwrite(pathname + "\\" + "mask" + "\\" + "ALL" + "\\" + os.path.splitext(path)[0].split("\\")[2] + "_mask" + ".png", background_mask)

  eng.quit()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('path', type=str)
  args = parser.parse_args()
  path = args.path
  print(path)
  main(path)