import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
import random
import torchvision.models as models
import torch
import torch.nn as nn
from models.RIAD.aug_dataset import CustomDataset

rgb_cap = cv2.VideoCapture('/home/quan/Desktop/company/dirty_dataset/rgb_video/1_rgb.avi')
mask_cap = cv2.VideoCapture('/home/quan/Desktop/company/dirty_dataset/rgb_video/1_mask.avi')

save_dir = '/home/quan/Desktop/company/dirty_dataset/RAID'
img_dir = os.path.join(save_dir, 'images')
mask_dir = os.path.join(save_dir, 'masks')

save_id = 0
while True:
    _, rgb_img = rgb_cap.read()
    _, mask_img = mask_cap.read()

    if (rgb_img is not None) and (mask_img is not None):
        h, w, c = rgb_img.shape

        show_img = np.zeros((h, w*2, 3), dtype=rgb_img.dtype)
        show_img[:, :w, :] = rgb_img
        show_img[:, w:, :] = mask_img

        cv2.imshow('ad', show_img)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(os.path.join(img_dir, '%d.jpg' % save_id), rgb_img)
            cv2.imwrite(os.path.join(mask_dir, '%d.jpg' % save_id), mask_img[:, :, 0])
            save_id += 1
