import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

from anomalib.config import get_configurable_parameters
from anomalib.deploy.inferencers.base import Inferencer
from anomalib.deploy.inferencers.torch import TorchInferencer
from anomalib.deploy.inferencers.openvino import OpenVINOInferencer

def parse_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--model_weight', type=str,
                        default='/home/quan/Desktop/company/dirty_dataset/test_img/output/patchcore/folder/weights/model.ckpt')
    parser.add_argument('--meta_data', type=str,
                        # default='/home/quan/Desktop/company/dirty_dataset/test_img/output/patchcore/folder/meta_data.json'
                        default=None
                        )
    parser.add_argument('--model_cfg', type=str,
                        default='/home/quan/Desktop/company/Reconstruct3D_Pipeline/models/abnormal_detect/cfg/patchcore_2.yaml')
    parser.add_argument('--image', type=str,
                        default='/home/quan/Desktop/company/dirty_dataset/test_img/test/img/2.jpg')
    parser.add_argument('--mask', type=str,
                        default='/home/quan/Desktop/company/dirty_dataset/test_img/test/mask/2.jpg')

    args = parser.parse_args()
    return args

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

def infer(infer_model: Inferencer, img):
    output = infer_model.predict(image=img, superimpose=False, overlay_mask=False)
    anomaly_map, score = output

    plt.imshow(anomaly_map)
    plt.show()

    # print(score)
    # print(anomaly_map.min(), anomaly_map.max())
    defect_mask = np.zeros(anomaly_map.shape, dtype=np.uint8)
    defect_mask[anomaly_map>=0.45] = 255

    return defect_mask, score

def main():
    args = parse_args()

    img = cv2.imread(args.image)
    mask = cv2.imread(args.mask, cv2.IMREAD_UNCHANGED)
    img, mask = post_process(rgb_img=img, label_img=mask)

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.equalizeHist(gray)
    # edge = cv2.Canny(gray, 100, 200)
    # cv2.imshow("mask image", edge)
    # cv2.imshow('gray', gray)
    # cv2.imshow('d', img)
    # cv2.waitKey(0)

    infer_model = load_infer_model(args)

    defect_mask, score = infer(infer_model, img.copy())

    # defect_mask, score = infer_model.predict(image=img, superimpose=False, overlay_mask=False)
    # plt.imshow(defect_mask)
    # plt.show()

    contours, hierarchy = cv2.findContours(defect_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c_id in range(len(contours)):
        cv2.drawContours(img, contours, c_id, (0, 0, 255), 2)
    cv2.putText(img, 'Score:%f' % score, org=(5, 20),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                color=(0, 0, 255), thickness=1)

    cv2.imshow('d', img)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
