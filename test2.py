import visdom
import os
import cv2
import time
import random
import numpy as np

src_dir = '/home/quan/Desktop/tempary/temp/good'
paths = os.listdir(src_dir)

vis = visdom.Visdom()
while True:
    path = random.choice(paths)
    img = cv2.imread(os.path.join(src_dir, path))
    img = np.transpose(img, (2, 0, 1))
    vis.image(img, win='img')
    time.sleep(1.0)
