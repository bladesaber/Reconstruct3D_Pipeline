import argparse
import os
import cv2
import torch
import numpy as np
import math
from typing import Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime

from models.RIAD.model_unet import UNet
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

def post_process(rgb_img, label_img):
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
        img, mask,
        cutout_size, num_disjoint_masks=3,
        with_normalize=True, channel_first=False,
):
    h, w, c = img.shape
    disjoint_masks = create_disjoint_masks(
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

def main_with_mask(fix_mask=False, run_count=1):
    args = parse_args()

    device = args.device
    time_tag = (datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")

    # network = UNet()
    network = UNet_Gan()
    weight = torch.load(args.weight)['state_dict']
    # load_weight = {}
    # for key in weight:
    #     if 'perceptual_loss_fn' not in key:
    #         load_weight[key] = weight[key]
    # weight = load_weight
    network.load_state_dict(weight)

    if device == 'cuda':
        network = network.to(torch.device('cuda:0'))
    network.eval()

    save_dir = os.path.join(args.save_dir, time_tag)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    cutout_sizes = (2, 4)
    for path in tqdm(os.listdir(args.img_dir)):
        img_path = os.path.join(args.img_dir, path)
        mask_path = os.path.join(args.mask_dir, path)

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        img, mask = post_process(img, mask)

        img = cv2.resize(img, (args.width, args.height))
        mask = cv2.resize(mask, (args.width, args.height))

        name = path.split('.')[0]
        name_dir = os.path.join(save_dir, name)
        if not os.path.exists(name_dir):
            os.mkdir(name_dir)
        cv2.imwrite(
            os.path.join(name_dir, 'orig.jpg'), img
        )

        for cutout_size in cutout_sizes:
            for count_id in range(run_count):
                if (count_id==0) or (not fix_mask):
                    imgs, disjoint_masks, disjoint_imgs = parse_img(
                        img=img, mask=mask,
                        cutout_size=cutout_size, num_disjoint_masks=3,
                        with_normalize=True, channel_first=True
                    )
                    # disjoint_imgs = disjoint_imgs[np.newaxis, ...]
                    # disjoint_imgs = torch.from_numpy(disjoint_imgs)
                    disjoint_masks = disjoint_masks[np.newaxis, ...]
                    disjoint_masks = torch.from_numpy(disjoint_masks)
                    imgs = imgs[np.newaxis, ...]
                    imgs = torch.from_numpy(imgs)

                    if device == 'cuda':
                        imgs = imgs.to(torch.device('cuda:0'))
                        disjoint_masks = disjoint_masks.to(torch.device('cuda:0'))
                        # disjoint_imgs = disjoint_imgs.to(torch.device('cuda:0'))

                ### debug
                # for mask_id in range(disjoint_masks.shape[0]):
                #     plt.figure('rgb')
                #     plt.imshow(img)
                #     plt.figure('mask')
                #     plt.imshow(disjoint_masks[mask_id, ...])
                #     plt.figure('mask_img')
                #     plt.imshow(disjoint_imgs[mask_id, ...])
                #     plt.show()

                fake_img = network.inferv2(disjoint_masks=disjoint_masks, imgs=imgs)
                rimg = fake_img.detach().cpu().numpy()[0, ...]
                rimg = np.transpose(rimg, (1, 2, 0))
                rimg = (rimg * 255.).astype(np.uint8)

                # plt.imshow(rimg)
                # plt.show()

                cv2.imwrite(
                    os.path.join(name_dir, 'cut%d_count%d.jpg' % (cutout_size, count_id)), rimg
                )

                imgs = fake_img

if __name__ == '__main__':
    main_with_mask(fix_mask=True, run_count=6)
