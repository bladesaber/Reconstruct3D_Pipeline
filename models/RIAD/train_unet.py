import argparse
import os

import cv2
from torch.utils.data import DataLoader
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import datetime
import numpy as np

from tensorboardX import SummaryWriter
from models.RIAD.aug_dataset import CustomDataset
from models.RIAD.model_unet import UNet
from models.utils.utils import Best_Saver

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')

    parser.add_argument('--experient', type=str, help='',
                        default='experiment_1')
    parser.add_argument('--save_dir', type=str, help='',
                        default='/home/quan/Desktop/tempary/output')
    parser.add_argument('--mask_dir', type=str, help='',
                        default='/home/quan/Desktop/company/dirty_dataset/RAID/masks')
    parser.add_argument('--img_dir', type=str, help='',
                        default='/home/quan/Desktop/company/dirty_dataset/RAID/images')

    parser.add_argument('--device', type=str, default='cpu')

    parser.add_argument('--optimizer_type', type=str, help='', default='Adam')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--minimum_lr', type=float, default=1e-4)
    parser.add_argument('--regularization', type=float, default=0.0005)
    parser.add_argument('--accumulate', type=int, default=1)
    parser.add_argument('--max_epoches', type=int, default=300)

    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr_update_patient', type=int, default=10)
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=240)

    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--checkpoint_interval', type=int, default=1)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    save_dir = os.path.join(args.save_dir, args.experient)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(os.path.join(save_dir, 'checkpoints')):
        os.mkdir(os.path.join(save_dir, 'checkpoints'))

    logger = SummaryWriter(log_dir=save_dir)
    vis_dir = os.path.join(save_dir, 'vis')
    if not os.path.exists(vis_dir):
        os.mkdir(vis_dir)

    network = UNet()
    dataset = CustomDataset(
        img_dir=args.img_dir,
        mask_dir=args.mask_dir,
        channel_first=True,
        with_aug=False,
        with_normalize=True,
        width=args.width, height=args.height,
        cutout_sizes=(2, 4), num_disjoint_masks=3
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.regularization)
    scheduler = ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=args.lr_update_patient,
        verbose=True, cooldown=0
    )

    device = args.device
    if device == 'cuda':
        network = network.to(torch.device('cuda:0'))

    time_tag = (datetime.datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    saver = Best_Saver(
        path=os.path.join(save_dir, 'checkpoints', "model_last.pth"),
        meta=time_tag
    )

    epoch = 0
    step = 0
    while True:
        network.train()

        current_lr = optimizer.param_groups[0]['lr']

        batch_mse_losses = []
        batch_ssim_losses = []
        batch_msgm_losses = []
        batch_total_losses = []
        for i, data_batch in enumerate(dataloader):
            step += 1

            batch_imgs, batch_masks, batch_mask_imgs, batch_obj_masks = data_batch
            if device == 'cuda':
                batch_imgs = batch_imgs.to(torch.device('cuda:0'))
                batch_masks = batch_masks.to(torch.device('cuda:0'))
                batch_mask_imgs = batch_mask_imgs.to(torch.device('cuda:0'))
                batch_obj_masks = batch_obj_masks.to(torch.device('cuda:0'))

            loss_dict = network.train_step(
                imgs=batch_imgs,
                disjoint_masks=batch_masks,
                mask_imgs=batch_mask_imgs,
                obj_masks=batch_obj_masks
            )

            total_loss = loss_dict['total']
            total_loss.backward()
            if step % args.accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()

            loss_mse_float = loss_dict['mse'].cpu().detach().item()
            loss_ssim_float = loss_dict['ssim'].cpu().detach().item()
            loss_msgm_float = loss_dict['msgm'].cpu().detach().item()
            loss_total_float = total_loss.cpu().detach().item()

            s = 'epoch:%d/step:%d lr:%1.7f loss:%5.5f mse:%.3f ssim:%.3f msgm:%.3f' % (
                epoch, step, current_lr, loss_total_float, loss_mse_float, loss_ssim_float, loss_msgm_float
            )
            print(s)

            batch_mse_losses.append(loss_mse_float)
            batch_ssim_losses.append(loss_ssim_float)
            batch_msgm_losses.append(loss_msgm_float)
            batch_total_losses.append(loss_total_float)

        rimgs = loss_dict['rimgs'].detach().cpu().numpy()
        rimgs = np.transpose(rimgs, (0, 2, 3, 1))
        rimgs = (rimgs * 255.).astype(np.uint8)
        batch_imgs_float = batch_imgs.detach().cpu().numpy()
        batch_imgs_float = np.transpose(batch_imgs_float, (0, 2, 3, 1))
        batch_imgs_float = (batch_imgs_float * 255.).astype(np.uint8)
        for rimg_id in range(rimgs.shape[0]):
            rimg = rimgs[rimg_id, ...]
            bimg = batch_imgs_float[rimg_id, ...]
            cv2.imwrite(os.path.join(vis_dir, "fake_epoch%d_id%d.jpg" % (epoch, rimg_id)), rimg)
            cv2.imwrite(os.path.join(vis_dir, "real_epoch%d_id%d.jpg" % (epoch, rimg_id)), bimg)

        cur_mse_loss = np.mean(batch_mse_losses)
        cur_ssim_loss = np.mean(batch_ssim_losses)
        cur_msgm_loss = np.mean(batch_msgm_losses)
        cur_total_loss = np.mean(batch_total_losses)

        batch_mse_losses.clear()
        batch_ssim_losses.clear()
        batch_msgm_losses.clear()
        batch_total_losses.clear()

        logger.add_scalar('mse_loss', cur_mse_loss, global_step=epoch)
        logger.add_scalar('ssim_loss', cur_ssim_loss, global_step=epoch)
        logger.add_scalar('msgm_loss', cur_msgm_loss, global_step=epoch)
        logger.add_scalar('total_loss', cur_total_loss, global_step=epoch)
        logger.add_scalar('lr', current_lr, global_step=epoch)
        print('###### epoch:%d lr:%1.7f loss:%5.5f mse:%.3f ssim:%.3f msgm:%.3f \n' %
              (epoch, current_lr, cur_total_loss, cur_mse_loss, cur_ssim_loss, cur_msgm_loss)
              )

        epoch += 1

        if (epoch > args.warmup) and (epoch % args.checkpoint_interval == 0):
            scheduler.step(cur_total_loss)
            saver.save(network, score=cur_total_loss, epoch=epoch)

            if current_lr < args.minimum_lr:
                break

        if epoch > args.max_epoches:
            break

    saver.save(network, score=cur_total_loss, epoch=epoch)

if __name__ == '__main__':
    main()
