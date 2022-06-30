import argparse
import torch
import numpy as np
import cv2


try:
    from gradcam_utils import GradCAM
    from model import Resnet18_model, Resnet50_model
except:
    from models.gradcam.gradcam_utils import GradCAM
    from models.gradcam.model import Resnet18_model, Resnet50_model
def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')

    parser.add_argument('--ckpt_path', type=str, help='',
                        default='/home/quan/Desktop/tempary/output/experiment_1/checkpoints/model_last.pth')
    parser.add_argument('--device', type=str, default='cpu')

    parser.add_argument('--rgb_avi', type=str,
                        default="/home/quan/Desktop/company/dirty_dataset/rgb_video/1_rgb.avi")
    parser.add_argument('--mask_avi', type=str,
                        default="/home/quan/Desktop/company/dirty_dataset/rgb_video/1_mask.avi")

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

def post_process(rgb_img, label_img):
    mask = np.zeros(label_img.shape, dtype=np.uint8)
    mask[label_img < 100] = 0
    mask[label_img > 140] = 0
    mask[np.bitwise_and(label_img < 140, label_img > 80)] = 1

    num_labels, labels, stats, centers = cv2.connectedComponentsWithStats(
        mask, connectivity=8, ltype=cv2.CV_32S
    )

    select_area, select_label = 0, -1
    for idx in range(1, num_labels, 1):
        x, y, w, h, area = stats[idx]

        if area > select_area:
            select_area = area
            select_label = idx

    mask = np.zeros(label_img.shape, dtype=np.uint8)
    mask[labels == select_label] = 255

    rgb_img[mask != 255, :] = 0

    return rgb_img, mask

normalized_dict = {
    'mean': np.array([123.675, 116.28, 103.53]),
    'std': np.array([58.395, 57.12, 57.375])
}
def imnormalize_(img, mean, std):
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    img = img.astype(np.float32)
    cv2.subtract(img, mean, img)
    cv2.multiply(img, stdinv, img)
    return img

def main():
    args = parse_args()

    rgb_cap = cv2.VideoCapture(args.rgb_avi)
    mask_cap = cv2.VideoCapture(args.mask_avi)

    # net = Resnet18_model(is_train=False)
    net = Resnet50_model(is_train=False, with_init=False)

    use_cuda = False
    if args.device == 'cuda':
        use_cuda = True
        net = net.cuda()
    net.eval()

    weight = torch.load(args.ckpt_path)['state_dict']
    net.load_state_dict(weight)

    target_layers = [net.backbone.layer4]
    cam = GradCAM(model=net, target_layers=target_layers, use_cuda=use_cuda)
    auto_mode = 0

    while True:
        _, rgb_img = rgb_cap.read()
        _, mask_img = mask_cap.read()

        if (rgb_img is not None) and (mask_img is not None):
            rgb_img = cv2.resize(rgb_img, (640, 480))
            mask_img = cv2.resize(mask_img, (640, 480))

            pose_rgb = rgb_img.copy()
            pose_rgb, mask = post_process(pose_rgb, label_img=mask_img[:, :, 0])
            temp_rgb = pose_rgb.copy()

            pose_rgb = imnormalize_(
                pose_rgb,
                mean=normalized_dict['mean'],
                std=normalized_dict['std']
            )
            pose_rgb = np.transpose(pose_rgb, (2, 0, 1))
            pose_rgb = (pose_rgb[np.newaxis, ...]).astype(np.float32)
            input_tensor = torch.from_numpy(pose_rgb)
            if args.device == 'cuda':
                input_tensor = input_tensor.cuda()

            out = net(input_tensor)
            out = out.cpu().detach().numpy()
            out = out[0]
            print('[DEBUG]: logits: good:%.2f bad:%.2f'%(out[0], out[1]))

            target_category = 1
            grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
            grayscale_cam = grayscale_cam[0, :]

            heatmap = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            # heatmap = np.tile(grayscale_cam[..., np.newaxis], (1, 1, 3))
            # heatmap = (heatmap *255.).astype(np.uint8)

            show_img = np.zeros((480, 1280, 3), dtype=np.uint8)
            show_img[:480, :640, :] = temp_rgb
            show_img[:480, 640:, :] = heatmap

            cv2.imshow('rgb', show_img)
            key = cv2.waitKey(auto_mode)
            if key == ord('q'):
                break
            elif key == ord('p'):
                auto_mode = 1
            elif key == ord('o'):
                auto_mode = 0
            else:
                pass

if __name__ == '__main__':
    main()
