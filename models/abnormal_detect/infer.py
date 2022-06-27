import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

from anomalib.config import get_configurable_parameters
from anomalib.deploy.inferencers.base import Inferencer
from anomalib.deploy.inferencers.torch import TorchInferencer
from anomalib.deploy.inferencers.openvino import OpenVINOInferencer

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--rgb_avi', type=str,
                        default="/home/quan/Desktop/tempary/test_dataset/test.mp4")
    parser.add_argument('--mask_avi', type=str,
                        default="/home/quan/Desktop/tempary/test_dataset/mask.avi")
    parser.add_argument('--save_dir', type=str,
                        default='/home/quan/Desktop/tempary/test_dataset/defect_data/good')

    parser.add_argument('--use_infer_model', type=int, default=1)
    parser.add_argument('--model_weight', type=str,
                        default='/home/quan/Desktop/tempary/test_dataset/output/patchcore/folder/weights/model.ckpt')
    parser.add_argument('--meta_data', type=str,
                        default=None)
    parser.add_argument('--model_cfg', type=str,
                        default='/home/quan/Desktop/company/Reconstruct3D_Pipeline/models/abnormal_detect/cfg/patchcore.yaml')

    args = parser.parse_args()
    return args

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

        if area>select_area:
            select_area=area
            select_label = idx

    mask = np.zeros(label_img.shape, dtype=np.uint8)
    mask[labels==select_label] = 255
    rgb_img[mask!=255, :] = 0

    return rgb_img, mask

def load_infer_model(args):
    config = get_configurable_parameters(config_path=args.model_cfg)

    extension = args.model_weight.split('.')[-1]
    if extension in ("ckpt"):
        inferencer = TorchInferencer(config=config, model_source=args.model_weight, meta_data_path=args.meta_data)

    elif extension in ("onnx", "bin", "xml"):
        inferencer = OpenVINOInferencer(config=config, path=args.model_weight, meta_data_path=args.meta_data)

    else:
        raise ValueError(
            f"Model extension is not supported. Torch Inferencer exptects a .ckpt file,"
            f"OpenVINO Inferencer expects either .onnx, .bin or .xml file. Got {extension}"
        )

    return inferencer

def infer(infer_model: Inferencer, img):
    # img = cv2.resize(img, (416, 416))

    output = infer_model.predict(image=img, superimpose=True, overlay_mask=True)
    anomaly_map, score = output
    cv2.imshow('d', anomaly_map)
    cv2.waitKey(0)

    raise ValueError

def main():
    background = 0
    car_label = 114
    window_label = 255

    args = parse_args()

    infer_model = None
    if args.use_infer_model>0:
        infer_model = load_infer_model(args)
    print(infer_model)

    rgb_cap = cv2.VideoCapture(args.rgb_avi)
    mask_cap = cv2.VideoCapture(args.mask_avi)

    save_id = 0
    while True:
        _, rgb_img = rgb_cap.read()
        _, mask_img = mask_cap.read()

        pose_rgb = rgb_img.copy()
        pose_rgb, mask = post_process(pose_rgb, label_img=mask_img[:, :, 0])

        if infer_model is not None:
            infer(infer_model, pose_rgb)

        rgb_img = cv2.resize(rgb_img, (640, 480))
        mask_img = cv2.resize(mask_img, (640, 480))
        pose_rgb = cv2.resize(pose_rgb, (640, 480))

        show_img = np.zeros((960, 1280, 3), dtype=np.uint8)
        show_img[:480, :640, :] = rgb_img
        show_img[:480, 640:, :] = mask_img
        show_img[480:, :640, :] = pose_rgb

        cv2.imshow('rgb', show_img)
        key = cv2.waitKey(0)
        if key==ord('q'):
            break
        if key == ord('s'):
            cv2.imwrite(os.path.join(args.save_dir, '%d.jpg' % save_id), pose_rgb)
            save_id += 1
        else:
            pass

if __name__ == '__main__':
    main()

