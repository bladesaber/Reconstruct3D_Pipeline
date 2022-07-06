import argparse
import os
from torch.utils.data import DataLoader
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import datetime
import numpy as np

from tensorboardX import SummaryWriter
from models.gradcam.model import Resnet18_model, Resnet50_model
from models.gradcam.aug_dataset import CutpasteDataset
from models.utils.utils import Last_Saver

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')

    parser.add_argument('--experient', type=str, help='',
                        default='experiment_3')
    parser.add_argument('--save_dir', type=str, help='',
                        default='/home/quan/Desktop/tempary/output')

    parser.add_argument('--img_dir', type=str,
                        default='/home/quan/Desktop/company/dirty_dataset/defect_data/good')
    parser.add_argument('--support_dir', type=str,
                        default='/home/quan/Desktop/company/support_dataset')

    parser.add_argument('--device', type=str, default='cpu')

    parser.add_argument('--optimizer_type', type=str, help='', default='Adam')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--minimum_lr', type=float, default=1e-4)
    parser.add_argument('--regularization', type=float, default=0.0005)
    parser.add_argument('--accumulate', type=int, default=1)
    parser.add_argument('--max_epoches', type=int, default=300)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr_update_patient', type=int, default=10)

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

    network = Resnet18_model()
    # network = Resnet50_model()
    dataset = CutpasteDataset(
        img_dir=args.img_dir,
        support_dir=args.support_dir,
        channel_first=True,
        with_normalize=False
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.regularization)
    scheduler = ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=args.lr_update_patient,
        verbose=True, cooldown=0
    )

    if device == 'cuda':
        network = network.to(torch.device('cuda:0'))

    time_tag = (datetime.datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    saver = Last_Saver(
        path=os.path.join(save_dir, 'checkpoints', "model_last.pth"),
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
                batch_img = batch_img.to(torch.device('cuda:0'))
                batch_labels = batch_labels.to(torch.device('cuda:0'))

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
        print('###### epoch:%d loss:%f acc:%f \n'%(epoch, curr_loss, curr_acc))

        epoch += 1

        if (epoch > args.warmup) and (epoch % args.checkpoint_interval == 0):
            scheduler.step(curr_loss)
            saver.save(network)

            if current_lr < args.minimum_lr:
                break

        if epoch > args.max_epoches:
            break

    saver.save(network)

if __name__ == '__main__':
    main()
