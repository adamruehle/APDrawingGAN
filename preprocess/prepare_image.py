import matlab.engine
import detect_facial_landmarks as dfl
import argparse
import os

def main(path):
  file = dfl.detect_facial_landmarks(path)
  eng = matlab.engine.start_matlab()
  function_directory = 'preprocess'
  eng.addpath(os.path.abspath(function_directory), nargout=0)
  landmarks = eng.load(file)
  (img, txt) = eng.face_align_512(path, landmarks, "imgs")
  eng.quit()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('path', type=str)
  args = parser.parse_args()
  path = args.path
  print(path)
  main(path)