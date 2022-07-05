import cv2
from torch.utils.data.dataset import Dataset
import os
import random
import numpy as np
from PIL import Image
import albumentations as albu
import imgaug.augmenters as iaa

class CutpasteDataset(Dataset):
    def __init__(
            self,
            img_dir, support_dir,
            wmin_ratio=0.1, wmax_ratio=0.3,
            hmin_ratio=0.1, hmax_ratio=0.3,
            with_normalize=True,
            width=640, height=480, channel_first=True,
    ):
        self.img_dir = img_dir
        self.paths = os.listdir(img_dir)
        self.num = len(self.paths)

        self.support_dir = support_dir
        self.support_paths = os.listdir(support_dir)

        self.img_width = width
        self.img_height = height
        self.channel_first = channel_first

        self.transformer_albu = self.get_albu_parser()
        self.transformer_imgaug = self.get_imgaug_parser()

        self.wmin_ratio = wmin_ratio
        self.wmax_ratio = wmax_ratio
        self.hmin_ratio = hmin_ratio
        self.hmax_ratio = hmax_ratio
        self.with_normalize = with_normalize

        self.normalized_dict = {
            'mean': np.array([123.675, 116.28, 103.53]),
            'std': np.array([58.395, 57.12, 57.375])
        }

    def __len__(self):
        return self.num

    def cv2pillow(self, img_bgr):
        return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    def pillow2cv(self, img_rgb):
        return cv2.cvtColor(np.asarray(img_rgb), cv2.COLOR_RGB2BGR)

    def crop_paste(self, image, rotation=(-45, 45), thresold=0.7):
        orig_image = image.copy()
        height, width, c = image.shape

        while True:
            image = orig_image.copy()

            patch_w = random.randint(int(width*self.wmin_ratio), int(width * self.wmax_ratio))
            patch_h = random.randint(int(height*self.hmin_ratio), int(height * self.hmax_ratio))
            area = patch_w * patch_h

            patch_left, patch_top = random.randint(0, width-patch_w), random.randint(0, height-patch_h)
            patch_right, patch_bottom = patch_left + patch_w, patch_top + patch_h

            crop_img = image[patch_top:patch_bottom, patch_left:patch_right, :]
            crop_img = self.cv2pillow(crop_img)

            random_rotate = random.uniform(*rotation)
            crop_img = crop_img.convert("RGBA").rotate(random_rotate, expand=True)
            mask = crop_img.split()[-1]

            paste_left, paste_top = random.randint(0, width-patch_w), random.randint(0, height-patch_h)
            image = self.cv2pillow(image)
            image.paste(crop_img, (paste_left, paste_top), mask=mask)

            image = self.pillow2cv(image)

            dif_map = np.abs(image - orig_image)
            dif_map = np.sum(dif_map, axis=2)
            dif_map = dif_map > 15.0
            dif_area = dif_map.sum()

            if dif_area>area*thresold:
                break

        return image

    def crop_paste_from_caltech(self, image, rotation=(-45, 45), thresold=0.7):
        orig_image = image.copy()
        height, width, c = image.shape

        while True:
            image = orig_image.copy()

            patch_w = random.randint(int(width * self.wmin_ratio), int(width * self.wmax_ratio))
            patch_h = random.randint(int(height * self.hmin_ratio), int(height * self.hmax_ratio))
            area = patch_w * patch_h

            select_path = random.choice(self.support_paths)
            select_path = os.path.join(self.support_dir, select_path)
            select_img = cv2.imread(select_path)
            sh, sw, sc = select_img.shape

            if patch_w<sw:
                patch_left = random.randint(0, sw - patch_w)
                patch_right = patch_left + patch_w
            else:
                patch_left = 0
                patch_right = sw

            if patch_h<sh:
                patch_top = random.randint(0, sh - patch_h)
                patch_bottom = patch_top + patch_h
            else:
                patch_top = 0
                patch_bottom = sh

            crop_img = select_img[patch_top:patch_bottom, patch_left:patch_right, :]
            crop_img = self.cv2pillow(crop_img)

            random_rotate = random.uniform(*rotation)
            crop_img = crop_img.convert("RGBA").rotate(random_rotate, expand=True)
            mask = crop_img.split()[-1]

            loc_left, loc_top = random.randint(0, width - patch_w), random.randint(0, height - patch_h)
            image = self.cv2pillow(image)
            image.paste(crop_img, (loc_left, loc_top), mask=mask)

            image = self.pillow2cv(image)

            dif_map = np.abs(image - orig_image)
            dif_map = np.sum(dif_map, axis=2)
            dif_map = dif_map > 15.0
            dif_area = dif_map.sum()

            if dif_area > area * thresold:
                break

        return image

    def get_albu_parser(self):
        transform = albu.Compose([
            albu.GaussianBlur(blur_limit=(1, 3), p=0.5),
            albu.HueSaturationValue(p=0.5),
            albu.RandomBrightness(p=0.5),
            albu.RandomContrast(p=0.5)
        ])
        return transform

    def get_imgaug_parser(self):
        noise_parser = iaa.AdditiveGaussianNoise(scale=(1.0, 10.0))
        transformer = iaa.Sequential([noise_parser])
        return transformer

    def imnormalize_(self, img, mean, std):
        mean = np.float64(mean.reshape(1, -1))
        stdinv = 1 / np.float64(std.reshape(1, -1))
        img = img.astype(np.float32)
        cv2.subtract(img, mean, img)
        cv2.multiply(img, stdinv, img)
        return img

    def __getitem__(self, index):
        index = index % self.num

        img_path = self.paths[index]
        img_path = os.path.join(self.img_dir, img_path)

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        transformed = self.transformer_albu(image=img)
        img = transformed["image"]

        if random.uniform(0.0, 1.0)>0.5:
            if random.uniform(0.0, 1.0)>0.2:
                img = self.crop_paste(img)
            else:
                img = self.crop_paste_from_caltech(img)
            label = 1
        else:
            label = 0

        if random.uniform(0.0, 1.0)>0.5:
            img = img[np.newaxis, ...]
            img = self.transformer_imgaug(images=img)
            img = img[0, ...]

        img = cv2.resize(img, (self.img_width, self.img_height))
        if self.with_normalize:
            img = self.imnormalize_(img, mean=self.normalized_dict['mean'], std=self.normalized_dict['std'])

        if self.channel_first:
            img = np.transpose(img, (2, 0, 1))

        return img.astype(np.float32), label

class Cutpaste_ObjDet_Dataset(Dataset):
    def __init__(
            self,
            img_dir, support_dir,
            wmin_ratio=0.1, wmax_ratio=0.3,
            hmin_ratio=0.1, hmax_ratio=0.3,
            with_normalize=True,
            width=640, height=480, channel_first=True,
    ):
        self.img_dir = img_dir
        self.paths = os.listdir(img_dir)
        self.num = len(self.paths)

        self.support_dir = support_dir
        self.support_paths = os.listdir(support_dir)

        self.img_width = width
        self.img_height = height
        self.channel_first = channel_first

        self.transformer_albu = self.get_albu_parser()
        self.transformer_imgaug = self.get_imgaug_parser()

        self.wmin_ratio = wmin_ratio
        self.wmax_ratio = wmax_ratio
        self.hmin_ratio = hmin_ratio
        self.hmax_ratio = hmax_ratio
        self.with_normalize = with_normalize

        self.normalized_dict = {
            'mean': np.array([123.675, 116.28, 103.53]),
            'std': np.array([58.395, 57.12, 57.375])
        }

    def __len__(self):
        return self.num

    def cv2pillow(self, img_bgr):
        return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    def pillow2cv(self, img_rgb):
        return cv2.cvtColor(np.asarray(img_rgb), cv2.COLOR_RGB2BGR)

    def crop_paste(self, image, rotation=(-45, 45), thresold=0.85):
        orig_image = image.copy()
        height, width, c = image.shape

        while True:
            image = orig_image.copy()
            patch_w = random.randint(int(width*self.wmin_ratio), int(width * self.wmax_ratio))
            patch_h = random.randint(int(height*self.hmin_ratio), int(height * self.hmax_ratio))
            area = patch_w * patch_h

            patch_left, patch_top = random.randint(0, width-patch_w), random.randint(0, height-patch_h)
            patch_right, patch_bottom = patch_left + patch_w, patch_top + patch_h

            crop_img = image[patch_top:patch_bottom, patch_left:patch_right, :]
            crop_img = self.cv2pillow(crop_img)

            random_rotate = random.uniform(*rotation)
            crop_img = crop_img.convert("RGBA").rotate(random_rotate, expand=True)
            mask = crop_img.split()[-1]

            paste_left, paste_top = random.randint(0, width-patch_w), random.randint(0, height-patch_h)
            image = self.cv2pillow(image)
            image.paste(crop_img, (paste_left, paste_top), mask=mask)

            image = self.pillow2cv(image)

            dif_map = np.abs(image - orig_image)
            dif_map = np.sum(dif_map, axis=2)
            dif_map = dif_map > 15.0
            dif_area = dif_map.sum()

            if dif_area>area*thresold:

                dif_map = dif_map.astype(np.uint8)

                ys, xs = np.where(dif_map==1)
                xmin = xs.min()
                xmax = xs.max()
                ymin = ys.min()
                ymax = ys.max()
                mask = dif_map

                break

        return image, (xmin, ymin, xmax, ymax), mask

    def crop_paste_from_caltech(self, image, rotation=(-45, 45), thresold=0.7):
        orig_image = image.copy()
        height, width, c = image.shape

        while True:
            image = orig_image.copy()
            patch_w = random.randint(int(width * self.wmin_ratio), int(width * self.wmax_ratio))
            patch_h = random.randint(int(height * self.hmin_ratio), int(height * self.hmax_ratio))
            area = patch_w * patch_h

            select_path = random.choice(self.support_paths)
            select_path = os.path.join(self.support_dir, select_path)
            select_img = cv2.imread(select_path)
            sh, sw, sc = select_img.shape

            if patch_w<sw:
                patch_left = random.randint(0, sw - patch_w)
                patch_right = patch_left + patch_w
            else:
                patch_left = 0
                patch_right = sw

            if patch_h<sh:
                patch_top = random.randint(0, sh - patch_h)
                patch_bottom = patch_top + patch_h
            else:
                patch_top = 0
                patch_bottom = sh

            crop_img = select_img[patch_top:patch_bottom, patch_left:patch_right, :]
            crop_img = self.cv2pillow(crop_img)

            random_rotate = random.uniform(*rotation)
            crop_img = crop_img.convert("RGBA").rotate(random_rotate, expand=True)
            mask = crop_img.split()[-1]

            loc_left, loc_top = random.randint(0, width - patch_w), random.randint(0, height - patch_h)
            image = self.cv2pillow(image)
            image.paste(crop_img, (loc_left, loc_top), mask=mask)

            image = self.pillow2cv(image)

            dif_map = np.abs(image - orig_image)
            dif_map = np.sum(dif_map, axis=2)
            dif_map = dif_map > 15.0
            dif_area = dif_map.sum()

            if dif_area > area * thresold:
                dif_map = dif_map.astype(np.uint8)
                ys, xs = np.where(dif_map == 1)
                xmin = xs.min()
                xmax = xs.max()
                ymin = ys.min()
                ymax = ys.max()
                mask = dif_map

                break

        return image, (xmin, ymin, xmax, ymax), mask

    def get_albu_parser(self):
        transform = albu.Compose([
            # albu.GaussianBlur(blur_limit=(1, 3), p=0.5),
            albu.HueSaturationValue(p=0.5),
            albu.RandomBrightness(p=0.5),
            # albu.RandomContrast(p=0.5)
        ])
        return transform

    def get_imgaug_parser(self):
        noise_parser = iaa.AdditiveGaussianNoise(scale=(1.0, 10.0))
        transformer = iaa.Sequential([noise_parser])
        return transformer

    def imnormalize_(self, img, mean, std):
        mean = np.float64(mean.reshape(1, -1))
        stdinv = 1 / np.float64(std.reshape(1, -1))
        img = img.astype(np.float32)
        cv2.subtract(img, mean, img)
        cv2.multiply(img, stdinv, img)
        return img

    def __getitem__(self, index):
        index = index % self.num

        img_path = self.paths[index]
        img_path = os.path.join(self.img_dir, img_path)

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        transformed = self.transformer_albu(image=img)
        img = transformed["image"]

        if random.uniform(0.0, 1.0)>0.2:
            img, pos, mask = self.crop_paste(img)
        else:
            img, pos, mask = self.crop_paste_from_caltech(img)

        # if random.uniform(0.0, 1.0)>0.5:
        #     img = img[np.newaxis, ...]
        #     img = self.transformer_imgaug(images=img)
        #     img = img[0, ...]

        img = cv2.resize(img, (self.img_width, self.img_height))
        if self.with_normalize:
            img = self.imnormalize_(img, mean=self.normalized_dict['mean'], std=self.normalized_dict['std'])

        if self.channel_first:
            img = np.transpose(img, (2, 0, 1))

        return img.astype(np.float32), pos, mask

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # dataset = CutpasteDataset(
    #     img_dir='/home/quan/Desktop/company/dirty_dataset/defect_data/good',
    #     support_dir='/home/quan/Desktop/company/support_dataset',
    #     channel_first=False, with_normalize=True
    # )
    #
    # labels = []
    # for idx, (img, label) in enumerate(dataset):
    #     print('[DEBUG]: ', label)
    #
    #     img = ((img-img.min())/(img.max() - img.min()) * 255.).astype(np.uint8)
    #
    #     plt.imshow(img)
    #     plt.show()
    #
    #     labels.append(label)
    #
    #     if idx>300:
    #         break
    #
    # labels = np.array(labels)
    # print('pos sum', (labels == 0).sum()/labels.shape[0])
    # print('neg sum', (labels == 1).sum() / labels.shape[0])

    dataset = Cutpaste_ObjDet_Dataset(
        img_dir='/home/quan/Desktop/company/dirty_dataset/defect_data/good',
        support_dir='/home/quan/Desktop/company/support_dataset',
        channel_first=False, with_normalize=False
    )

    for idx, (img, pos, mask) in enumerate(dataset):
        img = ((img - img.min()) / (img.max() - img.min()) * 255.).astype(np.uint8)

        xmin, ymin, xmax, ymax = pos
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(255, 0, 0), thickness=1)

        plt.figure('rgb')
        plt.imshow(img)
        plt.figure('mask')
        plt.imshow(mask)

        plt.show()
