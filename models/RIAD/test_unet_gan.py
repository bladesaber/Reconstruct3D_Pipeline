import argparse
import os
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import DataLoader
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import datetime
import numpy as np

import torch.nn.functional as F
from torchmetrics.functional import accuracy

from tensorboardX import SummaryWriter
from models.utils.utils import Best_Saver
from models.RIAD.model_unet_gan import UNet_Gan, Res_Discriminator
from models.RIAD.aug_dataset import CustomDataset
from models.RIAD.model_unet_gan import Adv_BCELoss_Trainer
from models.utils.logger import Metric_Recorder

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')

    parser.add_argument('--save_dir', type=str, help='',
                        default='/home/psdz/HDD/quan/output')
    parser.add_argument('--mask_dir', type=str, help='',
                        default='/home/psdz/HDD/quan/RAID/masks')
    parser.add_argument('--img_dir', type=str, help='',
                        default='/home/psdz/HDD/quan/RAID/images')
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--resume_generator_path', type=str,
                        default='/home/psdz/HDD/quan/output/experiment_1/checkpoints/model_unet.pth')
    parser.add_argument('--resume_discirmator_path', type=str,
                        default='/home/psdz/HDD/quan/output/experiment_1/checkpoints/model_discrimator.pth')

    args = parser.parse_args()
    return args

def test_discrimator():
    args = parse_args()

    device = args.device

    unet = UNet_Gan()
    unet_weight = torch.load(args.resume_generator_path)['state_dict']
    unet.load_state_dict(unet_weight)
    unet.eval()

    discrimator = Res_Discriminator(num_classes=2)
    dis_weight = torch.load(args.resume_discirmator_path)['state_dict']
    discrimator.load_state_dict(dis_weight)
    print('[DEBUG]: Loading weight %s Successfuly' % args.resume_discirmator_path)
    discrimator.eval()

    dataset = CustomDataset(
        img_dir=args.img_dir,
        mask_dir=args.mask_dir,
        channel_first=True,
        with_aug=False,
        with_normalize=True,
        width=args.width, height=args.height,
        cutout_sizes=(2, 4, 8), num_disjoint_masks=4
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    if device == 'cuda':
        unet = unet.to(torch.device('cuda:0'))
        discrimator = discrimator.to(torch.device('cuda:0'))

    wait_time = 0
    correct, wrong = 0.0, 0.0
    for _ in range(4):
        for i, data_batch in enumerate(dataloader):

            batch_imgs, batch_masks, batch_mask_imgs, batch_obj_masks = data_batch
            if device == 'cuda':
                batch_imgs = batch_imgs.to(torch.device('cuda:0'))
                batch_masks = batch_masks.to(torch.device('cuda:0'))
                batch_mask_imgs = batch_mask_imgs.to(torch.device('cuda:0'))
                # batch_obj_masks = batch_obj_masks.to(torch.device('cuda:0'))

            with torch.no_grad():
                fake_imgs = unet.infer(
                    disjoint_masks=batch_masks,
                    mask_imgs=batch_mask_imgs,
                )

            fake_results = discrimator.infer(fake_imgs)
            real_results = discrimator.infer(batch_imgs)

            for idx in range(batch_imgs.shape[0]):
                real_img = batch_imgs[idx, ...].detach().cpu().numpy()
                real_img = np.transpose(real_img, (1, 2, 0))
                fake_img = fake_imgs[idx, ...].detach().cpu().numpy()
                fake_img = np.transpose(fake_img, (1, 2, 0))

                real_res = real_results[idx, ...].cpu().detach().numpy()
                pred_real_id = np.argmax(real_res)
                fake_res = fake_results[idx, ...].cpu().detach().numpy()
                pred_fake_id = np.argmax(fake_res)

                if pred_real_id == 0:
                    correct += 1.0
                else:
                    wrong += 1.0

                if pred_fake_id == 1:
                    correct += 1.0
                else:
                    wrong += 1.0

                # print('Real Cate -- pred:%d label:%d'%(pred_real_id, 0))
                # print('Fake Cate -- pred:%d label:%d'%(pred_fake_id, 1))

                cv2.imshow('real', real_img)
                cv2.imshow('fake', fake_img)
                key = cv2.waitKey(wait_time)
                if key == ord('o'):
                    wait_time = 1
                elif key == ord('p'):
                    wait_time = 0
                else:
                    pass

    print('Correct: %f Wrong: %f' % (correct / (correct + wrong), wrong / (correct + wrong)))

if __name__ == '__main__':
    test_discrimator()
