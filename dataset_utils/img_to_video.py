import os
import cv2
import argparse

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dir', type=str,
                        default="/home/quan/Desktop/company/dirty_dataset/rgb_video/car")
    parser.add_argument('--save_path', type=str,
                        default="/home/quan/Desktop/company/dirty_dataset/rgb_video/3_mask.avi")
    parser.add_argument('--width', type=int, default=540)
    parser.add_argument('--height', type=int, default=960)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    path_dict = {}
    for path in os.listdir(args.dir):
        path_id = int(path.split('.')[0])
        path_dict[path_id] = os.path.join(args.dir, path)

    video_writter = cv2.VideoWriter(
            args.save_path,
            cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (args.width, args.height)
        )
    for idx in range(len(os.listdir(args.dir))):
        img = cv2.imread(path_dict[idx], cv2.IMREAD_UNCHANGED)

        if img.ndim==2:
            img = img[..., np.newaxis]

        if img.shape[-1] == 1:
            img = np.tile(img, (1, 1, 3))

        video_writter.write(img)

    video_writter.release()

if __name__ == '__main__':
    main()
