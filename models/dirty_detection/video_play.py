import cv2
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--avi', type=str,
                        default="/home/quan/Desktop/tempary/test_dataset/test.mp4")
    parser.add_argument('--save_dir', type=str,
                        default='/home/quan/Desktop/tempary/test_dataset/JPEG')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.avi)

    save_id = 0
    while True:
        ret, ori_img = cap.read()

        if ori_img is not None:
            img = cv2.resize(ori_img, (640, 480))

            cv2.imshow('debug', img)
            key = cv2.waitKey(0)
            if key == ord('s'):
                cv2.imwrite(os.path.join(args.save_dir, '%d.jpg'%save_id), img)
                save_id += 1
            elif key==ord('q'):
                break
            else:
                pass

if __name__ == '__main__':
    main()
