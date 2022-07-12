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
from models.RIAD.model_unet_gan import UNet_Gan, SN_Discriminator, RestNet_Discriminator, Res_Discriminator
from models.RIAD.aug_dataset import CustomDataset
from models.RIAD.model_unet_gan import Adv_BCELoss_Trainer
from models.utils.logger import Metric_Recorder

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')

    parser.add_argument('--experient', type=str, help='',
                        default='experiment_7')
    parser.add_argument('--save_dir', type=str, help='',
                        default='/home/psdz/HDD/quan/output')
    parser.add_argument('--mask_dir', type=str, help='',
                        default='/home/psdz/HDD/quan/RAID/masks')
    parser.add_argument('--img_dir', type=str, help='',
                        default='/home/psdz/HDD/quan/RAID/images')

    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--glr', type=float, default=0.01)
    parser.add_argument('--dlr', type=float, default=0.01)
    parser.add_argument('--regularization', type=float, default=0.0005)
    parser.add_argument('--gen_accumulate', type=int, default=1)
    parser.add_argument('--dis_accumulate', type=int, default=1)
    parser.add_argument('--max_epoches', type=int, default=100)

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)

    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--lr_update_patient', type=int, default=10)
    parser.add_argument('--checkpoint_interval', type=int, default=1)
    parser.add_argument('--minimum_lr', type=float, default=1e-4)

    parser.add_argument('--resume_generator_path', type=str,
                        default='/home/psdz/HDD/quan/output/experiment_6/checkpoints/model_unet.pth')
    parser.add_argument('--resume_discirmator_path', type=str,
                        default=None)

    args = parser.parse_args()
    return args

def train_discrimator():
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
    unet_weight = torch.load(args.resume_generator_path)['state_dict']
    unet.load_state_dict(unet_weight)
    unet.eval()

    discrimator = Res_Discriminator()

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

    disc_opt = torch.optim.Adam(discrimator.parameters(), lr=args.dlr, weight_decay=args.regularization)

    if device == 'cuda':
        unet = unet.to(torch.device('cuda:0'))
        discrimator = discrimator.to(torch.device('cuda:0'))

    time_tag = (datetime.datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    discrimator_saver = Best_Saver(
        path=os.path.join(save_dir, 'checkpoints', "model_discrimator.pth"),
        meta=time_tag
    )

    scheduler = ReduceLROnPlateau(
        disc_opt, 'min', factor=0.5, patience=args.lr_update_patient,
        verbose=True, cooldown=0
    )

    trainer = Adv_BCELoss_Trainer()
    recorder = Metric_Recorder()

    epoch = 0
    step = 0
    while True:
        discrimator.train()

        dis_lr = disc_opt.param_groups[0]['lr']

        for i, data_batch in enumerate(dataloader):
            step += 1

            batch_imgs, batch_masks, batch_mask_imgs, batch_obj_masks = data_batch
            if device == 'cuda':
                batch_imgs = batch_imgs.to(torch.device('cuda:0'))
                batch_masks = batch_masks.to(torch.device('cuda:0'))
                batch_mask_imgs = batch_mask_imgs.to(torch.device('cuda:0'))
                batch_obj_masks = batch_obj_masks.to(torch.device('cuda:0'))

            with torch.no_grad():
                fake_imgs = unet.infer(
                    disjoint_masks=batch_masks,
                    mask_imgs=batch_mask_imgs,
                )

            adv_loss_dict, adv_acc_dict = trainer.train(discrimator, batch_imgs, fake_imgs, with_gen_loss=False)
            dis_loss = adv_loss_dict['dis_loss']
            dis_acc = adv_acc_dict['dis_acc']

            dis_loss.backward()
            if step % args.dis_accumulate == 0:
                disc_opt.step()
                disc_opt.zero_grad()

            loss_dis_float = recorder.add_scalar_tensor('dis_loss', dis_loss)
            acc_dis_float = recorder.add_scalar_tensor('dis_acc', dis_acc)

            s = 'epoch:%d/step:%d lr:%1.7f dis_loss:%.3f dis_acc:%.3f' % (
                epoch, step, dis_lr, loss_dis_float, acc_dis_float
            )
            print(s)

        ### record image
        rimgs = fake_imgs.detach().cpu().numpy()
        rimgs = np.transpose(rimgs, (0, 2, 3, 1))
        rimgs = (rimgs * 255.).astype(np.uint8)
        rimg = rimgs[0, ...]
        cv2.imwrite(os.path.join(vis_dir, "fake_epoch%d_id%d.jpg" % (epoch, 0)), rimg)

        metric_dict = recorder.compute_mean()
        cur_dis_loss = metric_dict['dis_loss']
        recorder.clear()

        s = '###### epoch:%d ' % epoch
        for name in metric_dict.keys():
            logger.add_scalar(name, metric_dict[name], global_step=epoch)
            s += '%s:%.3f ' % (name, metric_dict[name])
        s += '\n'
        print(s)
        logger.add_scalar('lr', dis_lr, global_step=epoch)

        epoch += 1

        if (epoch > args.warmup) and (epoch % args.checkpoint_interval == 0):
            scheduler.step(cur_dis_loss)
            discrimator_saver.save(discrimator, score=cur_dis_loss, epoch=epoch)

            if dis_lr < args.minimum_lr:
                break

        if epoch > args.max_epoches:
            break

def train_unet():
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

    if device == 'cuda':
        unet = unet.to(torch.device('cuda:0'))

    time_tag = (datetime.datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    unet_saver = Best_Saver(
        path=os.path.join(save_dir, 'checkpoints', "model_unet.pth"),
        meta=time_tag
    )

    scheduler = ReduceLROnPlateau(
        unet_opt, 'min', factor=0.5, patience=args.lr_update_patient,
        verbose=True, cooldown=0
    )

    recorder = Metric_Recorder()

    epoch = 0
    step = 0
    while True:
        unet.train()

        gen_lr = unet_opt.param_groups[0]['lr']

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

            mse_loss = gen_loss_dict['mse']
            ssim_loss = gen_loss_dict['ssim']
            msgm_loss = gen_loss_dict['msgm']
            gen_loss = gen_loss_dict['total']

            gen_loss.backward()
            if step % args.gen_accumulate == 0:
                unet_opt.step()
                unet_opt.zero_grad()

            loss_mse_float = recorder.add_scalar_tensor('mse_loss', mse_loss)
            loss_ssim_float = recorder.add_scalar_tensor('ssim_loss', ssim_loss)
            loss_msgm_float = recorder.add_scalar_tensor('msgm_loss', msgm_loss)
            loss_gen_float = recorder.add_scalar_tensor('total_loss', gen_loss)

            s = 'epoch:%d/step:%d lr:%1.7f loss:%5.5f mse:%.3f ssim:%.3f msgm:%.3f' % (
                epoch, step, gen_lr, loss_gen_float, loss_mse_float, loss_ssim_float, loss_msgm_float
            )
            print(s)

        ### record image
        rimgs = fake_imgs.detach().cpu().numpy()
        rimgs = np.transpose(rimgs, (0, 2, 3, 1))
        rimgs = (rimgs * 255.).astype(np.uint8)
        batch_imgs_float = batch_imgs.detach().cpu().numpy()
        batch_imgs_float = np.transpose(batch_imgs_float, (0, 2, 3, 1))
        batch_imgs_float = (batch_imgs_float * 255.).astype(np.uint8)
        # for rimg_id in range(rimgs.shape[0]):
        #     rimg = rimgs[rimg_id, ...]
        #     bimg = batch_imgs_float[rimg_id, ...]
        #     cv2.imwrite(os.path.join(vis_dir, "fake_epoch%d_id%d.jpg" % (epoch, rimg_id)), rimg)
        #     cv2.imwrite(os.path.join(vis_dir, "real_epoch%d_id%d.jpg" % (epoch, rimg_id)), bimg)
        rimg = rimgs[0, ...]
        bimg = batch_imgs_float[0, ...]
        cv2.imwrite(os.path.join(vis_dir, "fake_epoch%d_id%d.jpg" % (epoch, 0)), rimg)
        cv2.imwrite(os.path.join(vis_dir, "real_epoch%d_id%d.jpg" % (epoch, 0)), bimg)

        metric_dict = recorder.compute_mean()
        cur_gen_loss = metric_dict['total_loss']
        recorder.clear()

        s = '###### epoch:%d '%epoch
        for name in metric_dict.keys():
            logger.add_scalar(name, metric_dict[name], global_step=epoch)
            s += '%s:%.3f '%(name, metric_dict[name])
        s += '\n'
        print(s)
        logger.add_scalar('lr', gen_lr, global_step=epoch)

        epoch += 1
        if (epoch > args.warmup) and (epoch % args.checkpoint_interval == 0):
            scheduler.step(cur_gen_loss)
            unet_saver.save(unet, score=cur_gen_loss, epoch=epoch)

            if gen_lr < args.minimum_lr:
                break

        if epoch > args.max_epoches:
            break

if __name__ == '__main__':
    # train_unet()
    train_discrimator()