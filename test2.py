import os
import shutil
import numpy as np
import torch

a = np.random.randint(0, 2, size=(2, 1, 5, 5)).astype(np.float32)
a = torch.from_numpy(a)

b = np.random.random(size=(2, 3, 5, 5)).astype(np.float32)
b = torch.from_numpy(b)

a = torch.tile(a, dims=(1, 3, 1, 1))
print(a.shape)
print(a)
print(b[a==1.0])
