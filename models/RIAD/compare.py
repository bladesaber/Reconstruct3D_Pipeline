import os
import math
from typing import Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse

from models.RIAD.loss_utils import MSGMSLoss
from models.RIAD.model_unet_gan import UNet_Gan

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')

    parser.add_argument('--weight', type=str, help='',
                        default='/home/psdz/HDD/quan/output/experiment_1/checkpoints/model_unet.pth')
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--mask_dir', type=str, help='',
                        default='/home/psdz/HDD/quan/output/test/mask')
    parser.add_argument('--img_dir', type=str, help='',
                        default='/home/psdz/HDD/quan/output/test/img')
    parser.add_argument('--save_dir', type=str,
                        default='/home/psdz/HDD/quan/output/test/result')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)

    args = parser.parse_args()
    return args

class AnoDetector(object):
    def __init__(self, args):
        self.device = args.device

        self.network = UNet_Gan()
        weight = torch.load(args.weight)['state_dict']
        self.network.load_state_dict(weight)
        if self.device == 'cuda':
            self.network = self.network.to(torch.device('cuda:0'))
        self.network.eval()

        self.cutout_sizes = (2, )
        self.run_count = 6
        self.args = args

        self.loss_fn = MSGMSLoss(num_scales=3)

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

    def parse_img(
            self,
            img, mask,
            cutout_size, num_disjoint_masks=3,
            with_normalize=True, channel_first=False,
    ):
        h, w, c = img.shape
        disjoint_masks = self.create_disjoint_masks(
            (h, w),
            cutout_size=cutout_size,
            num_disjoint_masks=num_disjoint_masks
        )

        disjoint_imgs = []
        # random_color = np.random.randint(0, 255, (3,))
        fill_color = np.array([0, 0, 0])
        for disjoint_id in range(num_disjoint_masks):
            disjoint_mask = disjoint_masks[disjoint_id, :, :]
            disjoint_img = img * disjoint_mask[..., np.newaxis]
            # disjoint_img[obj_mask == 0, :] = random_color
            disjoint_img[mask == 0, :] = fill_color

            disjoint_mask[mask == 0] = 1
            disjoint_masks[disjoint_id, :, :] = disjoint_mask

            disjoint_imgs.append(disjoint_img)
        disjoint_imgs = np.array(disjoint_imgs)

        if channel_first:
            img = np.transpose(img, (2, 0, 1))
            disjoint_imgs = np.transpose(disjoint_imgs, (0, 3, 1, 2))

        if with_normalize:
            img = img / 255.
            disjoint_imgs = disjoint_imgs / 255.

        return img.astype(np.float32), disjoint_masks.astype(np.float32), disjoint_imgs.astype(np.float32)

    def proprocess(self, img, img_mask):
        if img_mask.ndim == 3:
            img_mask = img_mask[:, :, 0]

        img, img_mask = self.post_process(img, img_mask)
        img = cv2.resize(img, (self.args.width, self.args.height))
        img_mask = cv2.resize(img_mask, (self.args.width, self.args.height))

        return img, img_mask

    def infer(self, img, disjoint_masks):
        disjoint_masks = disjoint_masks[np.newaxis, ...]
        disjoint_masks = torch.from_numpy(disjoint_masks)
        img = img[np.newaxis, ...]
        img = torch.from_numpy(img)

        if self.device == 'cuda':
            img = img.to(torch.device('cuda:0'))
            disjoint_masks = disjoint_masks.to(torch.device('cuda:0'))

        fake_img = self.network.inferv2(disjoint_masks=disjoint_masks, imgs=img)
        rimg = fake_img.detach().cpu().numpy()[0, ...]
        rimg = np.transpose(rimg, (1, 2, 0))
        rimg = (rimg * 255.).astype(np.uint8)

        return rimg

    def next_mask_img(self, orig_img, fake_img):
        orig_img = orig_img.astype(np.float32)
        fake_img = fake_img.astype(np.float32)

        orig_img_norm = (orig_img - orig_img.mean()) / orig_img.std()
        fake_img_norm = (fake_img - fake_img.mean()) / fake_img.std()

        img_dif = np.abs(orig_img_norm - fake_img_norm)
        img_dif = np.mean(img_dif, axis=2)
        img_dif_norm = (img_dif - img_dif.mean()) / img_dif.std()

        ano_mask = np.zeros(img_dif.shape)
        ano_mask[img_dif_norm>2.5] = 1.0

        # plt.figure('orig')
        # plt.imshow(ano_mask)

        # se = np.ones((3, 3), dtype=np.uint8)
        # # ano_mask = cv2.erode(ano_mask, se, None, (-1, -1), 1)
        # ano_mask = cv2.dilate(ano_mask, se, None, (-1, -1), 1)

        # plt.figure('after')
        # plt.imshow(ano_mask)
        # plt.show()

        fake_img_next = orig_img.copy()
        fake_img_next[ano_mask==1] = fake_img[ano_mask==1]
        fake_img_next = fake_img_next.astype(np.uint8)

        return fake_img_next, ano_mask

    def norm_and_channel_first(self, img):
        img = img.astype(np.float32)
        img = np.transpose(img, (2, 0, 1))
        img = img /255.

        return img

    def anno_detect(self, orig_img, fake_img):
        orig_img_tensor = (np.transpose(orig_img, (2, 0, 1))[np.newaxis, ...]).astype(np.float32) / 255.
        fake_img_tensor = (np.transpose(fake_img, (2, 0, 1))[np.newaxis, ...]).astype(np.float32) / 255.

        orig_img_tensor = torch.from_numpy(orig_img_tensor)
        fake_img_tensor = torch.from_numpy(fake_img_tensor)

        ano_mask = self.loss_fn(orig_img_tensor, fake_img_tensor, as_loss=False)
        ano_mask = ano_mask.cpu().numpy()
        ano_mask = np.transpose(ano_mask[0, ...], (1, 2, 0))
        ano_mask = ano_mask[:, :, 0]

        ano_score = ano_mask.max()

        ano_mask = (ano_mask - ano_mask.min()) / (ano_mask.max() - ano_mask.min())
        defect_mask = np.zeros(ano_mask.shape, dtype=np.uint8)
        defect_mask[ano_mask > 0.5] = 1

        return ano_score, defect_mask

    def show_image(self, img, ano_score, ano_mask):
        show_img = img.copy()
        contours, hierarchy = cv2.findContours(ano_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c_id in range(len(contours)):
            cv2.drawContours(show_img, contours, c_id, (0, 0, 255), 1)
        cv2.putText(show_img, 'Score:%f' % ano_score, org=(5, 20),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                    color=(0, 0, 255), thickness=1)
        return show_img

def main():
    args = parse_args()

    img_path = '/home/psdz/HDD/quan/output/test/img/11.jpg'
    mask_path = '/home/psdz/HDD/quan/output/test/mask/11.jpg'

    img = cv2.imread(img_path)
    img_mask = cv2.imread(mask_path)

    model = AnoDetector(args=args)
    img, img_mask = model.proprocess(img, img_mask)
    img = cv2.GaussianBlur(img, ksize=(3, 3), sigmaX=1, sigmaY=1)

    _, disjoint_masks, _ = model.parse_img(
        img=img, mask=img_mask,
        cutout_size=2, num_disjoint_masks=3,
        with_normalize=True, channel_first=True
    )

    run_count = 0
    img_infer = img.copy()
    ref_img = img.copy()
    while True:
        run_count += 1

        img_tensor = model.norm_and_channel_first(img_infer)
        fake_img = model.infer(img=img_tensor, disjoint_masks=disjoint_masks)
        r_fake_img, amask = model.next_mask_img(orig_img=ref_img, fake_img=fake_img)

        img_infer = r_fake_img

        cv2.imshow('orig', ref_img)
        cv2.imshow('amask', amask)
        cv2.imshow('fake', fake_img)
        cv2.imshow('rfake', r_fake_img)
        key = cv2.waitKey(0)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break
        else:
            pass

    ano_score, ano_map = model.anno_detect(orig_img=img, fake_img=r_fake_img)
    show_img = model.show_image(img, ano_score, ano_map)

    cv2.imshow('d', ano_map)
    cv2.imshow('result', show_img)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
