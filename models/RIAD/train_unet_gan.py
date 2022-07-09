import argparse
import os

import cv2
from torch.utils.data import DataLoader
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import datetime
import numpy as np

from tensorboardX import SummaryWriter
from models.utils.utils import Best_Saver
from models.RIAD.model_unet_gan import UNet_Gan, SN_Discriminator, RestNet_Discriminator
from models.RIAD.aug_dataset import CustomDataset
from models.RIAD.model_unet_gan import CustomTrainer

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')

    parser.add_argument('--experient', type=str, help='',
                        default='experiment_5')
    parser.add_argument('--save_dir', type=str, help='',
                        default='/home/psdz/HDD/quan/output')
    parser.add_argument('--mask_dir', type=str, help='',
                        default='/home/psdz/HDD/quan/RAID/masks')
    parser.add_argument('--img_dir', type=str, help='',
                        default='/home/psdz/HDD/quan/RAID/images')

    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--glr', type=float, default=0.0001)
    parser.add_argument('--dlr', type=float, default=0.01)
    parser.add_argument('--regularization', type=float, default=0.0005)
    parser.add_argument('--gen_accumulate', type=int, default=10000)
    parser.add_argument('--dis_accumulate', type=int, default=1)
    parser.add_argument('--max_epoches', type=int, default=500)

    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)

    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--checkpoint_interval', type=int, default=1)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    device = args.device

    save_dir = os.path.join(args.save_dir, args.experient)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(os.path.join(save_dir, 'checkpoints')):
        os.mkdir(os.path.join(save_dir, 'checkpoints'))

    logger = SummaryWriter(log_dir=save_dir)
    vis_dir = os.path.join(save_dir, 'vis')
    if not os.path.exists(vis_dir):
        os.mkdir(vis_dir)

    unet = UNet_Gan()
    # discrimator = SN_Discriminator(width=args.width, height=args.height)
    discrimator = RestNet_Discriminator()

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

    unet_opt = torch.optim.Adam(unet.parameters(), lr=args.glr, weight_decay=args.regularization)
    disc_opt = torch.optim.Adam(discrimator.parameters(), lr=args.dlr, weight_decay=args.regularization)

    if device == 'cuda':
        unet = unet.to(torch.device('cuda:0'))
        discrimator = discrimator.to(torch.device('cuda:0'))

    time_tag = (datetime.datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    unet_saver = Best_Saver(
        path=os.path.join(save_dir, 'checkpoints', "model_unet.pth"),
        meta=time_tag
    )
    discrimator_saver = Best_Saver(
        path=os.path.join(save_dir, 'checkpoints', "model_discrimator.pth"),
        meta=time_tag
    )

    trainer = CustomTrainer()

    epoch = 0
    step = 0
    while True:
        unet.train()
        discrimator.train()

        batch_mse_losses = []
        batch_ssim_losses = []
        batch_msgm_losses = []
        batch_adv_gen_losses = []
        batch_gen_losses = []
        batch_dis_losses = []
        batch_gen_acces = []
        batch_dis_acces = []
        for i, data_batch in enumerate(dataloader):
            step += 1

            batch_imgs, batch_masks, batch_mask_imgs, batch_obj_masks = data_batch
            if device == 'cuda':
                batch_imgs = batch_imgs.to(torch.device('cuda:0'))
                batch_masks = batch_masks.to(torch.device('cuda:0'))
                batch_mask_imgs = batch_mask_imgs.to(torch.device('cuda:0'))
                batch_obj_masks = batch_obj_masks.to(torch.device('cuda:0'))

            gen_loss_dict, fake_imgs = unet.train_step(
                imgs=batch_imgs,
                disjoint_masks=batch_masks,
                mask_imgs=batch_mask_imgs,
                obj_masks=batch_obj_masks
            )

            adv_loss_dict, adv_acc_dict = trainer.train(discrimator, batch_imgs, fake_imgs)

            mse_loss = gen_loss_dict['mse']
            ssim_loss = gen_loss_dict['ssim']
            msgm_loss = gen_loss_dict['msgm']

            dis_loss = adv_loss_dict['dis_loss']
            adv_gen_loss = adv_loss_dict['gen_loss']

            dis_acc = adv_acc_dict['dis_acc']
            gen_acc = adv_acc_dict['gen_acc']

            gen_loss = mse_loss + ssim_loss + msgm_loss + adv_gen_loss
            gen_loss.backward()
            dis_loss.backward()

            if step % args.gen_accumulate == 0:
                unet_opt.step()
                unet_opt.zero_grad()

            if step % args.dis_accumulate == 0:
                disc_opt.step()
                disc_opt.zero_grad()

            loss_mse_float = mse_loss.cpu().detach().item()
            loss_ssim_float = ssim_loss.cpu().detach().item()
            loss_msgm_float = msgm_loss.cpu().detach().item()
            loss_adv_gen_float = adv_gen_loss.cpu().detach().item()
            loss_gen_float = gen_loss.cpu().detach().item()
            loss_dis_float = dis_loss.cpu().detach().item()

            gen_acc_float = gen_acc.cpu().item()
            dis_acc_float = dis_acc.cpu().item()

            s = 'epoch:%d/step:%d gen_loss:%3.3f dis_loss:%3.3f mse:%.3f ssim:%.3f msgm:%.3f advGen_loss:%.3f ' \
                'dis_acc:%.1f gen_acc:%.1f'% (
                epoch, step, loss_gen_float, loss_dis_float,
                loss_mse_float, loss_ssim_float, loss_msgm_float, loss_adv_gen_float,
                dis_acc_float, gen_acc_float
            )
            print(s)

            ### record
            batch_mse_losses.append(loss_mse_float)
            batch_ssim_losses.append(loss_ssim_float)
            batch_msgm_losses.append(loss_msgm_float)
            batch_adv_gen_losses.append(loss_adv_gen_float)
            batch_gen_losses.append(loss_gen_float)
            batch_dis_losses.append(loss_dis_float)

            batch_gen_acces.append(gen_acc_float)
            batch_dis_acces.append(dis_acc_float)

        rimgs = fake_imgs.detach().cpu().numpy()
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
        cur_adv_gen_loss = np.mean(batch_adv_gen_losses)
        cur_gen_loss = np.mean(batch_gen_losses)
        cur_dis_loss = np.mean(batch_dis_losses)
        cur_gen_acc = np.mean(batch_gen_acces)
        cur_dis_acc = np.mean(batch_dis_acces)

        batch_mse_losses.clear()
        batch_ssim_losses.clear()
        batch_msgm_losses.clear()
        batch_adv_gen_losses.clear()
        batch_gen_losses.clear()
        batch_dis_losses.clear()
        batch_gen_acces.clear()
        batch_dis_acces.clear()

        logger.add_scalar('mse_loss', cur_mse_loss, global_step=epoch)
        logger.add_scalar('ssim_loss', cur_ssim_loss, global_step=epoch)
        logger.add_scalar('msgm_loss', cur_msgm_loss, global_step=epoch)
        logger.add_scalar('adv_gen_loss', cur_adv_gen_loss, global_step=epoch)
        logger.add_scalar('gen_loss', cur_gen_loss, global_step=epoch)
        logger.add_scalar('dis_loss', cur_dis_loss, global_step=epoch)
        logger.add_scalar('gen_acc', cur_gen_acc, global_step=epoch)
        logger.add_scalar('dis_acc', cur_dis_acc, global_step=epoch)
        print('###### epoch:%d gen_loss:%3.3f dis_loss:%3.3f mse:%.3f ssim:%.3f msgm:%.3f advGen_loss:msgm:%.3f ' \
                'gen_acc:%.2f dis_acc:%.2f \n' %
              (epoch, cur_gen_loss,
               cur_dis_loss, cur_mse_loss, cur_ssim_loss, cur_msgm_loss,
               cur_adv_gen_loss, cur_gen_acc, cur_dis_acc)
              )

        epoch += 1

        if (epoch > args.warmup) and (epoch % args.checkpoint_interval == 0):
            unet_saver.save(unet, score=cur_gen_loss, epoch=epoch)
            discrimator_saver.save(discrimator, score=cur_dis_loss, epoch=epoch)

        if epoch > args.max_epoches:
            break

if __name__ == '__main__':
    main()
