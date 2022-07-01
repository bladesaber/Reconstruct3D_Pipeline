import os
from torch.utils.data import Dataset
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from typing import Tuple
import math
import random

class CustomDataset(Dataset):
    def __init__(
            self,
            img_dir,
            num_disjoint_masks=3, cutout_sizes=(4, 8, 16),
            with_normalize=True, width=640, height=480, channel_first=True,
    ):
        self.img_dir = img_dir
        self.paths = os.listdir(img_dir)
        self.num = len(self.paths)

        self.transformer_albu = self.get_albu_parser()

        self.img_width = width
        self.img_height = height
        self.channel_first = channel_first
        self.with_normalize = with_normalize

        self.cutout_sizes = cutout_sizes
        self.num_disjoint_masks = num_disjoint_masks

    def __len__(self):
        return self.num

    def get_albu_parser(self):
        transform = albu.Compose([
            # albu.GaussianBlur(blur_limit=(1, 3), p=0.5),
            albu.HueSaturationValue(p=0.5),
            # albu.RandomBrightness(p=0.5),
            # albu.RandomContrast(p=0.5)
            albu.Normalize(),
            # ToTensorV2()
        ])
        return transform

    def __getitem__(self, index):
        index = index % self.num

        img_path = self.paths[index]
        img_path = os.path.join(self.img_dir, img_path)

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        transformed = self.transformer_albu(image=img)
        img = transformed["image"]

        img = cv2.resize(img, (self.img_width, self.img_height))

        h, w, c = img.shape
        cutout_size = random.choice(self.cutout_sizes)
        disjoint_masks = self.create_disjoint_masks(
            (h, w),
            cutout_size=cutout_size,
            num_disjoint_masks=self.num_disjoint_masks
        )

        if self.channel_first:
            img = np.transpose(img, (2, 0, 1))

        return img.astype(np.float32), disjoint_masks.astype(np.float32)

    def re_normalized(self, img):
        rimg = ((img-img.min())/(img.max()-img.min()) * 255.).astype(np.uint8)
        return rimg

    def create_disjoint_masks(
        self,
        img_size: Tuple[int, int],
        cutout_size: int = 8,
        num_disjoint_masks: int = 3,
    ):
        img_h, img_w = img_size
        grid_h = math.ceil(img_h / cutout_size)
        grid_w = math.ceil(img_w / cutout_size)
        num_grids = grid_h * grid_w

        disjoint_masks = []
        for grid_ids in np.array_split(np.random.permutation(num_grids), num_disjoint_masks):
            flatten_mask = np.ones(num_grids)
            flatten_mask[grid_ids] = 0
            mask = flatten_mask.reshape((grid_h, grid_w))
            mask = mask.repeat(cutout_size, axis=0).repeat(cutout_size, axis=1)
            disjoint_masks.append(mask)

        disjoint_masks = np.array(disjoint_masks)

        return disjoint_masks

