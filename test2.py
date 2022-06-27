import visdom
import os
import cv2
import time
import random
import numpy as np
import albumentations as albu
import torchvision
from albumentations.pytorch.transforms import ToTensorV2
import torch

w = torch.load('/home/quan/Desktop/tempary/test_dataset/output/patchcore/folder/weights/model.ckpt')
print(w.keys())