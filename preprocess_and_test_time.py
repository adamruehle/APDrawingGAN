import time
import os


start_time = time.time()
os.system("python preprocess/prepare_image.py myImages/AdamRuehle/AdamRuehle.png")
os.system("python test.py --dataroot dataset/data/test_single --name formal_author --model test --dataset_mode single --norm batch --use_local --which_epoch 300 --gpu_ids -1")


end_time = time.time()
print("Total time for preprocessing and image generation (seconds):", end_time - start_time)