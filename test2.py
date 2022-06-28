import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology

cap1 = cv2.VideoCapture("/home/quan/Desktop/company/dirty_dataset/rgb_video/1_rgb.avi")
cap2 = cv2.VideoCapture("/home/quan/Desktop/company/dirty_dataset/rgb_video/2_rgb.avi")

while True:
    _, img1 = cap1.read()
    _, img2 = cap2.read()

    if (img1 is not None) and (img2 is not None):
        img1 = cv2.resize(img1, (640, 480))
        img2 = cv2.resize(img2, (640, 480))

        show_img = np.zeros((480, 1280, 3), dtype=np.uint8)
        show_img[:480, :640, :] = img1
        show_img[:480, 640:, :] = img2

        cv2.imshow('rgb', show_img)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break