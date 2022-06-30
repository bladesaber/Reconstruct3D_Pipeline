import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

from models.gradcam.aug_dataset import CutpasteDataset
from models.gradcam.model import Resnet18_model, Resnet50_model

from models.gradcam.gradcam_utils import GradCAM
# from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')

    parser.add_argument('--ckpt_path', type=str, help='',
                        default='/home/quan/Desktop/tempary/output/experiment_1/checkpoints/model_last.pth')

    parser.add_argument('--img_dir', type=str,
                        default='/home/quan/Desktop/company/dirty_dataset/defect_data/good')
    parser.add_argument('--support_dir', type=str,
                        default='/home/quan/Desktop/company/support_dataset')
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()
    return args

def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def main():
    args = parse_args()

    dataset = CutpasteDataset(
        img_dir='/home/quan/Desktop/company/dirty_dataset/defect_data/good',
        support_dir='/home/quan/Desktop/company/support_dataset',
        channel_first=True, with_normalize=True
    )

    # net = Resnet18_model(is_train=False)
    net = Resnet50_model(is_train=False, with_init=False)
    if args.device == 'cuda':
        net = net.cuda()
    net.eval()

    weight = torch.load(args.ckpt_path)['state_dict']
    net.load_state_dict(weight)

    target_layers = [net.backbone.layer4[-1]]
    cam = GradCAM(model=net, target_layers=target_layers, use_cuda=False)
    # cam = GradCAM(model=net, target_layers=target_layers)

    for idx, (img, label) in enumerate(dataset):

        rgb_img = img.copy()
        rgb_img = np.transpose(rgb_img, (1, 2, 0))
        rgb_img = ((rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min()) * 255.).astype(np.uint8)

        img = img[np.newaxis, ...]
        img = torch.from_numpy(img)
        if args.device == 'cuda':
            img = img.cuda()

        out = net(img)
        out = out.cpu().detach().numpy()

        pred_label = np.argmax(out, axis=1)
        pred_label = pred_label[0]

        print('[DEBUG]: logits: ', out)
        print('[DEBUG]: Correct Label: %d pred_label: %d'%(label, pred_label))

        grayscale_cam = cam(input_tensor=img, target_category=1)
        # grayscale_cam = cam(input_tensor=img, targets=None,)
        grayscale_cam = grayscale_cam[0, :]
        heatmap = show_cam_on_image(rgb_img.copy(), grayscale_cam, use_rgb=True)

        plt.imshow(heatmap)
        plt.show()

if __name__ == '__main__':
    main()
