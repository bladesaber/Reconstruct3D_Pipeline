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
            img_dir, mask_dir,
            num_disjoint_masks=3, cutout_sizes=(2, 4, 8),
            with_normalize=True, with_aug=False,
            width=640, height=480, channel_first=True,
    ):
        self.img_dir = img_dir
        self.paths = os.listdir(img_dir)
        self.num = len(self.paths)
        self.mask_dir = mask_dir

        self.transformer_albu = self.get_albu_parser()

        self.img_width = width
        self.img_height = height
        self.channel_first = channel_first
        self.with_normalize = with_normalize
        self.with_aug = with_aug

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
            # albu.Normalize(),
            # ToTensorV2()
        ])
        return transform

    def __getitem__(self, index):
        index = index % self.num

        path = self.paths[index]
        img_path = os.path.join(self.img_dir, path)
        obj_mask_path = os.path.join(self.mask_dir, path)

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        obj_multi_mask = cv2.imread(obj_mask_path, cv2.IMREAD_UNCHANGED)
        img, _ = self.post_process(img, obj_multi_mask)

        obj_mask = np.zeros(obj_multi_mask.shape)
        obj_mask[np.bitwise_and(obj_multi_mask>100, obj_multi_mask<128)] = 1

        if self.with_aug:
            transformed = self.transformer_albu(image=img)
            img = transformed["image"]

        img = cv2.resize(img, (self.img_width, self.img_height))
        obj_mask = cv2.resize(obj_mask, (self.img_width, self.img_height))

        h, w, c = img.shape
        cutout_size = random.choice(self.cutout_sizes)
        disjoint_masks = self.create_disjoint_masks(
            (h, w),
            cutout_size=cutout_size,
            num_disjoint_masks=self.num_disjoint_masks
        )

        disjoint_imgs = []
        # random_color = np.random.randint(0, 255, (3,))
        fill_color = np.array([0, 0, 0])
        for disjoint_id in range(self.num_disjoint_masks):
            disjoint_mask = disjoint_masks[disjoint_id, :, :]
            disjoint_img = img * disjoint_mask[..., np.newaxis]
            # disjoint_img[obj_mask == 0, :] = random_color
            disjoint_img[obj_mask == 0, :] = fill_color

            disjoint_mask[obj_mask == 0] = 1
            disjoint_masks[disjoint_id, :, :] = disjoint_mask

            disjoint_imgs.append(disjoint_img)
        disjoint_imgs = np.array(disjoint_imgs)

        if self.channel_first:
            img = np.transpose(img, (2, 0, 1))
            disjoint_imgs = np.transpose(disjoint_imgs, (0, 3, 1, 2))

        if self.with_normalize:
            img = img / 255.
            disjoint_imgs = disjoint_imgs / 255.

        return img.astype(np.float32), disjoint_masks.astype(np.float32), disjoint_imgs.astype(np.float32)

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

    def post_process(self, rgb_img, label_img):
        mask = np.zeros(label_img.shape, dtype=np.uint8)
        mask[label_img < 100] = 0
        mask[label_img > 128] = 0
        mask[np.bitwise_and(label_img < 140, label_img > 80)] = 1

        num_labels, labels, stats, centers = cv2.connectedComponentsWithStats(
            mask, connectivity=8, ltype=cv2.CV_32S
        )

        select_area, select_label = 0, -1
        for idx in range(1, num_labels, 1):
            x, y, w, h, area = stats[idx]

            if area > select_area:
                select_area = area
                select_label = idx

        mask = np.zeros(label_img.shape, dtype=np.uint8)
        mask[labels == select_label] = 255

        rgb_img[mask != 255, :] = 0

        return rgb_img, mask

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dataset = CustomDataset(
        img_dir='/home/quan/Desktop/company/dirty_dataset/RAID/images',
        mask_dir='/home/quan/Desktop/company/dirty_dataset/RAID/masks',
        num_disjoint_masks=4, channel_first=False, with_aug=False, with_normalize=True
    )

    for img, masks, mask_imgs in dataset:

        for mimg_id in range(mask_imgs.shape[0]):
            mimg = mask_imgs[mimg_id, ...]
            mask = masks[mimg_id, ...]

            plt.figure('rgb')
            plt.imshow(img)
            plt.figure('%d_img'%mimg_id)
            plt.imshow(mimg)
            plt.figure('%d_mask' % mimg_id)
            plt.imshow(mask)

            plt.show()

    # from torch.utils.data import DataLoader
    #
    # dataset = CustomDataset(
    #     img_dir='/home/quan/Desktop/company/dirty_dataset/RAID/images',
    #     mask_dir='/home/quan/Desktop/company/dirty_dataset/RAID/masks',
    #     num_disjoint_masks=4, channel_first=True, with_normalize=False
    # )
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    # for img, masks, mask_imgs in dataloader:
    #     print(img.shape)
    #     print(masks.shape)
    #     print(mask_imgs.shape)
    #     break

