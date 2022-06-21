import pandas as pd
import numpy as np
import random
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

save_dir = '/home/quan/Desktop/tempary/temp/good'
src_dir = '/home/quan/Desktop/tempary/car_seg_data'
bad_dir = '/home/quan/Desktop/tempary/temp/bad'

# record = []

good_idx = 0
bad_idx = 0
ptr = 0

img_paths = os.listdir(src_dir)[:10000]
num_paths = len(img_paths)
while True:
    path = img_paths[ptr]
    img_path = os.path.join(src_dir, path)
    img = cv2.imread(img_path)

    h, w, c = img.shape
    # record.append([h, w, c])

    img = cv2.resize(img, (352, 256))

    cv2.imshow('d', img)
    key = cv2.waitKey(0)
    if key == ord('q'):
        break

    elif key == ord('s'):
        cv2.imwrite(os.path.join(save_dir, '%d.jpg' % good_idx), img)
        good_idx += 1
        ptr += 1

    elif key == ord('p'):
        cv2.imwrite(os.path.join(bad_dir, '%d.jpg' % bad_idx), img)
        bad_idx += 1
        ptr += 1

    elif key == ord('1'):
        ptr += 1
    elif key == ord('2'):
        ptr -= 1
    else:
        ptr += 1

    ptr = ptr % num_paths

# record = np.load('/home/quan/Desktop/tempary/temp/1.npy')
#
# select_bool = record[:, 0]<257
# record = record[select_bool]
#
# plt.hist(record[:, 1])
# plt.show()