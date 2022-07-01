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
from models.utils.utils import Best_Saver

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')

    parser.add_argument('--experient', type=str, help='',
                        default='experiment_2')
    parser.add_argument('--save_dir', type=str, help='',
                        default='/home/quan/Desktop/tempary/output')

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

    save_dir = os.path.join(args.save_dir, args.experient)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(os.path.join(save_dir, 'checkpoints')):
        os.mkdir(os.path.join(save_dir, 'checkpoints'))

    logger = SummaryWriter(log_dir=save_dir)


