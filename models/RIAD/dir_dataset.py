import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
import random
import cv2

class Dir_Dataset(Dataset):
    def __init__(self, dir, width, height, channel_first=True):
        self.dir = dir

        self.num = 0
        self.labels = os.listdir(dir)
        self.label_dir = {}
        self.label_paths = {}
        for label in self.labels:
            self.label_dir[label] = os.path.join(self.dir, label)
            self.label_paths[label] = os.listdir(self.label_dir[label])
            self.num += len(self.label_paths[label])

        self.label_to_id = {}
        for idx, label in enumerate(self.labels):
            self.label_to_id[label] = idx

        self.channel_first = channel_first
        self.width = width
        self.height = height

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        select_label = random.choice(self.labels)
        select_path = random.choice(self.label_paths[select_label])

        select_path = os.path.join(self.label_dir[select_label], select_path)
        img = cv2.imread(select_path)
        img = cv2.resize(img, (self.width, self.height))
        img = img.astype(np.float32)

        img = img /255.
        if self.channel_first:
            img = np.transpose(img, (2, 0, 1))

        label_id = self.label_to_id[select_label]

        return img, label_id, select_label

if __name__ == '__main__':
    dataset = Dir_Dataset(dir='/home/psdz/HDD/data/flower_data/train', channel_first=False, width=640, height=480)
    print(len(dataset))
    # for img, label_id, label in dataset:
    #     print(label, label_id)
    #     plt.imshow(img)
    #     plt.show()
