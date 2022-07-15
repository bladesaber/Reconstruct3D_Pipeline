import argparse
import os
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import DataLoader
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import datetime
import numpy as np

from tensorboardX import SummaryWriter
from models.utils.utils import Best_Saver
from models.RIAD.model_unet_gan import SN_Discriminator, RestNet_Discriminator, Res_Discriminator
from models.RIAD.dir_dataset import Dir_Dataset
from models.utils.logger import Metric_Recorder

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')

    parser.add_argument('--experient', type=str, help='',
                        default='dis1')
    parser.add_argument('--save_dir', type=str, help='',
                        default='/home/psdz/HDD/quan/output')
    parser.add_argument('--data_dir', type=str, help='',
                        default='/home/psdz/HDD/quan/temp_trash/fakeVSreal')

    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--dlr', type=float, default=0.001)
    parser.add_argument('--regularization', type=float, default=0.0005)
    parser.add_argument('--accumulate', type=int, default=1)
    parser.add_argument('--max_epoches', type=int, default=100)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)

    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--lr_update_patient', type=int, default=10)
    parser.add_argument('--checkpoint_interval', type=int, default=1)
    parser.add_argument('--minimum_lr', type=float, default=1e-4)

    parser.add_argument('--resume_weight', type=str,
                        default='/home/psdz/HDD/quan/output/experiment_1/checkpoints/model_discrimator.pth')

    args = parser.parse_args()
    return args

def train():
    args = parse_args()

    device = args.device

    save_dir = os.path.join(args.save_dir, args.experient)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(os.path.join(save_dir, 'checkpoints')):
        os.mkdir(os.path.join(save_dir, 'checkpoints'))

    logger = SummaryWriter(log_dir=save_dir)

    network = Res_Discriminator(num_classes=2, selfTrain=True)

    dataset = Dir_Dataset(
        dir=args.data_dir,
        channel_first=True,
        width=args.width, height=args.height,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(network.parameters(), lr=args.dlr, weight_decay=args.regularization)

    if device == 'cuda':
        network = network.to(torch.device('cuda:0'))

    time_tag = (datetime.datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    saver = Best_Saver(
        path=os.path.join(save_dir, 'checkpoints', "model_dis.pth"),
        meta=time_tag
    )

    scheduler = ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=args.lr_update_patient,
        verbose=True, cooldown=0
    )

    recorder = Metric_Recorder()

    epoch = 0
    step = 0
    while True:
        network.train()

        current_lr = optimizer.param_groups[0]['lr']

        for i, data_batch in enumerate(dataloader):
            step += 1

            batch_imgs, batch_label_ids, batch_labels = data_batch
            if device == 'cuda':
                batch_imgs = batch_imgs.to(torch.device('cuda:0'))
                batch_label_ids = batch_label_ids.to(torch.device('cuda:0'))
                # batch_labels = batch_labels.to(torch.device('cuda:0'))

            loss, acc = network.train_step(
                x=batch_imgs, labels=batch_label_ids
            )

            loss.backward()
            if step % args.accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()

            loss_float = recorder.add_scalar_tensor('loss', loss)
            acc_float = recorder.add_scalar_tensor('acc', acc)

            s = 'epoch:%d/step:%d loss:%5.5f lr:%1.7f acc:%f' % (epoch, step, loss_float, current_lr, acc_float)
            print(s)

        metric_dict = recorder.compute_mean()
        cur_loss = metric_dict['loss']
        recorder.clear()

        s = '###### epoch:%d '%epoch
        for name in metric_dict.keys():
            logger.add_scalar(name, metric_dict[name], global_step=epoch)
            s += '%s:%.3f '%(name, metric_dict[name])
        s += '\n'
        print(s)
        logger.add_scalar('lr', current_lr, global_step=epoch)

        epoch += 1
        if (epoch > args.warmup) and (epoch % args.checkpoint_interval == 0):
            scheduler.step(cur_loss)
            saver.save(network, score=cur_loss, epoch=epoch)

            if current_lr < args.minimum_lr:
                break

        if epoch > args.max_epoches:
            break

def infer():
    args = parse_args()

    device = args.device

    network = Res_Discriminator(num_classes=2, selfTrain=True)
    weight = torch.load(args.resume_weight)['state_dict']
    network.load_state_dict(weight)
    network.eval()

    dataset = Dir_Dataset(
        dir=args.data_dir,
        channel_first=True,
        width=args.width, height=args.height,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    if device == 'cuda':
        network = network.to(torch.device('cuda:0'))

    correct, wrong = 0.0, 0.0
    for i, data_batch in enumerate(dataloader):

        batch_imgs, batch_label_ids, batch_labels = data_batch
        if device == 'cuda':
            batch_imgs = batch_imgs.to(torch.device('cuda:0'))
            batch_label_ids = batch_label_ids.to(torch.device('cuda:0'))
            # batch_labels = batch_labels.to(torch.device('cuda:0'))

        results = network.infer(x=batch_imgs)
        for idx in range(batch_label_ids.shape[0]):
            img = batch_imgs[idx, ...].detach().cpu().numpy()
            img = np.transpose(img, (1, 2, 0))

            res = results[idx, ...].cpu().detach().numpy()
            pred_id = np.argmax(res)
            label_id = batch_label_ids[idx].cpu().item()

            if pred_id == label_id:
                correct += 1.0
            else:
                wrong += 1.0

            # print('pred id: ', pred_id, res)
            # print('Label: ', batch_label_ids[idx].cpu().item(), batch_labels[idx])
            # plt.imshow(img)
            # plt.show()

    print('Correct: %f Wrong: %f'%(correct/(correct+wrong), wrong/(correct+wrong)))


if __name__ == '__main__':
    # train()
    infer()
