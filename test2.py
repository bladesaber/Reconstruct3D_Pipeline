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

model = models.resnet50(pretrained=True)
# for name, param in model.named_parameters():
#     print(name)
print(model.layer4[-1])