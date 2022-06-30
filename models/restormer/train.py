import argparse
import os
import mlflow
from mlflow import log_metric, log_param, log_artifact
import yaml
import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad
import numpy as np
import time

from models.restormer.restormer import Restormer
from models.utils.utils import setup_optimizers
from models.restormer.dataset_utils import Dataset_PairedImage
from models.restormer.dataloader_utils import Group_DataLoader
from models.utils.utils import Last_Saver
from models.utils.logger import Logger_Visdom

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')

    parser.add_argument('--experient', type=str, help='',
                        default='experiment_1')
    parser.add_argument('--config_path', type=str, help='',
                        default='/home/quan/Desktop/company/Reconstruct3D_Pipeline/models/restormer/test_config.yaml')
    parser.add_argument('--save_dir', type=str, help='',
                        default='/home/quan/Desktop/tempary/output')
    parser.add_argument('--device', type=str, default='cpu')

    parser.add_argument('--optimizer_type', type=str, help='', default='Adam')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--minimum_lr', type=float, default=1e-5)
    parser.add_argument('--regularization', type=float, default=0.0005)
    parser.add_argument('--accumulate', type=int, default=1)
    parser.add_argument('--max_epoches', type=int, default=300)

    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--lr_update_patient', type=int, default=10)

    parser.add_argument('--clip_grad', type=int, default=0)
    parser.add_argument('--max_norm', default=35, type=int)
    parser.add_argument('--norm_type', default=2, type=int)

    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--checkpoint_interval', type=int, default=1)

    args = parser.parse_args()
    return args

def re_normalize(img):
    img_min = np.min(img, axis=(0, 1), keepdims=True)
    img_max = np.max(img, axis=(0, 1), keepdims=True)
    img =((img - img_min)/(img_max-img_min) * 255.).astype(np.uint8)

    return img

def train():
    args = parse_args()

    device = args.device

    with mlflow.start_run(run_name=args.experient, nested=True):
        log_artifact(args.config_path)
        log_param('device', device)
        log_param('lr', args.lr)
        log_param('minimum_lr', args.minimum_lr)
        log_param('optimizer_type', args.optimizer_type)
        log_param('regularization', args.regularization)
        log_param('batch_size', args.batch_size)
        log_param('lr_update_patient', args.lr_update_patient)
        log_param('accumulate', args.accumulate)
        log_param('clip-grad', args.clip_grad)
        log_param('max_norm', args.max_norm)
        log_param('norm_type', args.norm_type)
        log_param('warmup', args.warmup)

        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        if not os.path.exists(os.path.join(args.save_dir, 'checkpoints')):
            os.mkdir(os.path.join(args.save_dir, 'checkpoints'))

        with open(args.config_path, 'r') as f:
            config = yaml.load(f, yaml.FullLoader)

        network_cfg = config['network']
        network = Restormer(
            input_dim=network_cfg['input_dim'],
            out_dim=network_cfg['out_dim'],
            embed_dim=network_cfg['embed_dim'],
            num_blocks=network_cfg['num_blocks'],
            heads=network_cfg['heads'],
            ffn_expansion_factor=network_cfg['ffn_expansion_factor'],
            bias=network_cfg['bias'],
            dual_pixel_task=network_cfg['dual_pixel_task']
        )

        if device=='cuda':
            network = network.cuda()

        dataset_cfg = config['dataset']
        dataset = Dataset_PairedImage(img_dir=config['jpg_path'], config=dataset_cfg)
        dataLoader = Group_DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

        optimizer = setup_optimizers(
            network, opt_type=args.optimizer_type, lr=args.lr, regularization=args.regularization
        )
        scheduler = ReduceLROnPlateau(
            optimizer, 'min', factor=0.5, patience=args.lr_update_patient,
            verbose=True, cooldown=0
        )

        time_tag = (datetime.datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
        saver = Last_Saver(
            path=os.path.join(args.save_dir, 'checkpoints', "model_last.pth"),
            meta=time_tag
        )
        vidsom_logger = Logger_Visdom()

        log_time = time.time()

        epoch = 0
        step = 0
        while True:
            epoch += 1
            network.train()

            current_lr = optimizer.param_groups[0]['lr']
            log_metric('lr', current_lr, step=epoch)

            batch_losses = []
            for i, data_batch in enumerate(dataLoader.dataloader):
                step += 1

                noise_images, gt_images, names = data_batch
                if device=='cuda':
                    noise_images = noise_images.cuda()
                    gt_images = gt_images.cuda()

                loss, rimgs = network.train_step(noise_img=noise_images, gt_img=gt_images)

                if time.time() - log_time>1.0:
                    validate_img = rimgs.cpu().detach().numpy()
                    validate_img = np.transpose(validate_img, (0, 2, 3, 1))

                    noise_images_np = noise_images.cpu().detach().numpy()
                    noise_images_np = np.transpose(noise_images_np, (0, 2, 3, 1))

                    select_idx = np.random.choice(np.arange(0, validate_img.shape[0], 1), size=4)

                    for name_idx in select_idx:
                        vidsom_logger.log_img(re_normalize(validate_img[name_idx, ...]),
                                              name='reconstract_%d'%name_idx)
                        vidsom_logger.log_img(re_normalize(noise_images_np[name_idx, ...]),
                                              name='noise_%d' % name_idx)

                    log_time = time.time()

                loss.backward()
                if step % args.accumulate == 0:
                    if args.clip_grad:
                        clip_grad.clip_grad_norm_(network.parameters(), max_norm=args.max_norm, norm_type=args.norm_type)

                    optimizer.step()
                    optimizer.zero_grad()

                loss_float = loss.cpu().item()
                batch_losses.append(loss_float)

                s = 'epoch:%d/step:%d loss:%5.5f lr:%1.7f' % (epoch, step, loss_float, current_lr)
                print(s)

            batch_losses = np.array(batch_losses)
            curr_loss = batch_losses.mean()
            log_metric('loss', curr_loss, step=epoch)

            if (epoch > args.warmup) and (epoch % args.checkpoint_interval == 0):
                scheduler.step(curr_loss)
                saver.save(network)

                if current_lr < args.minimum_lr:
                    break

            if epoch > args.max_epoches:
                break

if __name__ == '__main__':
    train()
