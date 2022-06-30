import argparse
import os
from torch.utils.data import DataLoader
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import datetime
import numpy as np

from tensorboardX import SummaryWriter
from models.gradcam.model import Resnet18_model
from models.gradcam.aug_dataset import CutpasteDataset
from models.utils.utils import Last_Saver

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')

    parser.add_argument('--experient', type=str, help='',
                        default='experiment_1')
    parser.add_argument('--save_dir', type=str, help='',
                        default='/home/quan/Desktop/tempary/output')
    parser.add_argument('--device', type=str, default='cpu')

    parser.add_argument('--optimizer_type', type=str, help='', default='Adam')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--minimum_lr', type=float, default=1e-4)
    parser.add_argument('--regularization', type=float, default=0.0005)
    parser.add_argument('--accumulate', type=int, default=1)
    parser.add_argument('--max_epoches', type=int, default=300)

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr_update_patient', type=int, default=10)

    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--checkpoint_interval', type=int, default=1)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    device = args.device

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, 'checkpoints')):
        os.mkdir(os.path.join(args.save_dir, 'checkpoints'))

    logger = SummaryWriter(log_dir=args.save_dir)

    network = Resnet18_model()
    dataset = CutpasteDataset(
        img_dir='/home/quan/Desktop/company/dirty_dataset/defect_data/good',
        support_dir='/home/quan/Desktop/company/support_dataset',
        channel_first=True
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.regularization)
    scheduler = ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=args.lr_update_patient,
        verbose=True, cooldown=0
    )

    if device == 'cuda':
        network = network.cuda()

    time_tag = (datetime.datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    saver = Last_Saver(
        path=os.path.join(args.save_dir, 'checkpoints', "model_last.pth"),
        meta=time_tag
    )

    epoch = 0
    step = 0
    while True:
        network.train()

        current_lr = optimizer.param_groups[0]['lr']

        batch_losses = []
        batch_acc = []
        for i, data_batch in enumerate(dataloader):
            step += 1

            batch_img, batch_labels = data_batch
            if device == 'cuda':
                batch_img = batch_img.cuda()
                batch_labels = batch_labels.cuda()

            loss, acc = network.train_step(x=batch_img, labels=batch_labels)

            loss.backward()
            if step % args.accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()

            loss_float = loss.cpu().item()
            acc_float = acc.cpu().item()
            batch_losses.append(loss_float)
            batch_acc.append(acc_float)

            s = 'epoch:%d/step:%d loss:%5.5f lr:%1.7f acc:%f' % (epoch, step, loss_float, current_lr, acc_float)
            print(s)

        batch_losses = np.array(batch_losses)
        curr_loss = batch_losses.mean()
        batch_acc = np.array(batch_acc)
        curr_acc = batch_acc.mean()
        logger.add_scalar('loss', curr_loss, global_step=epoch)
        logger.add_scalar('lr', current_lr, global_step=epoch)
        logger.add_scalar('acc', curr_acc, global_step=epoch)
        print('###### epoch:%d loss:%f acc:%f'%(epoch, curr_loss, curr_acc))

        epoch += 1

        if (epoch > args.warmup) and (epoch % args.checkpoint_interval == 0):
            scheduler.step(curr_loss)
            saver.save(network)

            if current_lr < args.minimum_lr:
                break

        if epoch > args.max_epoches:
            break

if __name__ == '__main__':
    main()
