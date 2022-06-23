import visdom
import os
import cv2
import time
import random
import numpy as np
import albumentations as albu

transform = albu.Compose([
    albu.RandomCrop(width=256, height=256),
    albu.HorizontalFlip(p=0.5),
    albu.RandomBrightnessContrast(p=0.2),
])
