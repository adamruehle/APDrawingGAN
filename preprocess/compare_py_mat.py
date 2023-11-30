import prepare_mask
import matlab.engine
import os
import detect_facial_landmarks as dfl
import time
import face_align_512
import scipy

path = "myImages\\AdamRuehle\\AdamRuehle.png"
function_directory = "preprocess"
def test_matlab_function():
  start_time = time.time()
  file = dfl.detect_facial_landmarks(path)
  eng = matlab.engine.start_matlab()
  eng.addpath(os.path.abspath(function_directory), nargout=0)
  landmarks = eng.load(file)
  eng.face_align_512(path, landmarks, "dataset/data/test_single", "dataset/landmark/ALL")
  eng.quit()
  end_time = time.time() 
  return end_time - start_time

def test_python_function():
  start_time = time.time()
  file = dfl.detect_facial_landmarks(path)
  landmarks = scipy.io.loadmat(file)
  aligned_image_path = face_align_512.face_align_512(path, landmarks, "dataset/data/test_single", "dataset/landmark/ALL")
  end_time = time.time()
  return end_time - start_time


if __name__ == "__main__":
  m_end_time = test_matlab_function()
  p_end_time = test_python_function()
  print("Matlab implementation:", m_end_time)
  print("Python implementation:", p_end_time)
