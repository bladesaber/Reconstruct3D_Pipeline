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
                        default="/home/quan/Desktop/company/dirty_dataset/rgb_video/2_rgb.avi")
    parser.add_argument('--mask_avi', type=str,
                        default="/home/quan/Desktop/company/dirty_dataset/rgb_video/2_mask.avi")
    parser.add_argument('--save_dir', type=str,
                        default='/home/quan/Desktop/company/dirty_dataset/defect_data/record'
                        )

    parser.add_argument('--use_infer_model', type=int, default=1)
    parser.add_argument('--model_weight', type=str,
                        default='/home/quan/Desktop/company/dirty_dataset/output/patchcore/folder/weights/model.ckpt')
    parser.add_argument('--meta_data', type=str,
                        default='/home/quan/Desktop/company/dirty_dataset/output/patchcore/folder/meta_data.json')
    parser.add_argument('--model_cfg', type=str,
                        default='/home/quan/Desktop/company/Reconstruct3D_Pipeline/models/abnormal_detect/cfg/patchcore.yaml')
    parser.add_argument('--record_avi', type=int, default=1)
    parser.add_argument('--record_avi_path', type=str,
                        default='/home/quan/Desktop/company/dirty_dataset/result2.avi')

    args = parser.parse_args()
    return args

### --------------------------------------------------------------------------
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
    img = cv2.resize(img, (640, 480))

    output = infer_model.predict(image=img, superimpose=False, overlay_mask=False)
    anomaly_map, score = output

    # print(score)
    # print(anomaly_map.min(), anomaly_map.max())
    defect_mask = np.zeros(anomaly_map.shape, dtype=np.uint8)
    defect_mask[anomaly_map>=0.5] = 255

    return defect_mask, score

def main():
    background = 0
    car_label = 114
    window_label = 255

    args = parse_args()

    infer_model = None
    if args.use_infer_model>0:
        infer_model = load_infer_model(args)

    rgb_cap = cv2.VideoCapture(args.rgb_avi)
    mask_cap = cv2.VideoCapture(args.mask_avi)

    if args.record_avi>0:
        writer_avi = cv2.VideoWriter(
            args.record_avi_path,
            cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (1280, 960)
        )

    auto_mode = 0
    save_id = 36
    while True:
        _, rgb_img = rgb_cap.read()
        _, mask_img = mask_cap.read()

        if (rgb_img is not None) and (mask_img is not None):
            pose_rgb = rgb_img.copy()
            pose_rgb, mask = post_process(pose_rgb, label_img=mask_img[:, :, 0])

            if infer_model is not None:
                defect_mask, score = infer(infer_model, pose_rgb)
                # print('[DEBUG]###: Score: %f'%score)

            rgb_img = cv2.resize(rgb_img, (640, 480))
            mask_img = cv2.resize(mask_img, (640, 480))
            pose_rgb = cv2.resize(pose_rgb, (640, 480))

            ### debug draw picture
            show_img = np.zeros((960, 1280, 3), dtype=np.uint8)
            show_img[:480, :640, :] = rgb_img
            show_img[:480, 640:, :] = mask_img

            if infer_model is not None:
                show_img[480:, 640:, :] = np.tile(defect_mask[..., np.newaxis], (1, 1, 3))

                contours, hierarchy = cv2.findContours(defect_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for c_id in range(len(contours)):
                    cv2.drawContours(pose_rgb, contours, c_id, (0, 0, 255), 2)
                cv2.putText(show_img, 'Score:%f'%score, org=(5, 20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                            color=(0, 0, 255), thickness=2)

                # ### just for debug
                # if len(contours)>0:
                #     cv2.imwrite(os.path.join(args.save_dir, '%d.jpg' % save_id), pose_rgb)
                #     save_id += 1

            show_img[480:, :640, :] = pose_rgb

            if args.record_avi>0:
                writer_avi.write(show_img)

            cv2.imshow('rgb', show_img)
            key = cv2.waitKey(auto_mode)
            if key==ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(os.path.join(args.save_dir, '%d.jpg' % save_id), pose_rgb)
                save_id += 1
            elif key==ord('p'):
                auto_mode = 1
            elif key==ord('o'):
                auto_mode = 0
            else:
                pass

    if args.record_avi > 0:
        writer_avi.release()

if __name__ == '__main__':
    main()

