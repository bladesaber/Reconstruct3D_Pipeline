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

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512, 64)
print(model)
# for idx, i in enumerate(model.children()):
#     print(idx)
#     print(i)

# a = torch.from_numpy(np.random.random((2, 3, 480, 640)).astype(np.float32))
# c = model(a)
# print(c.shape)