import argparse
import pycocotools.coco as coco
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import random

def parse_args():
    parser = argparse.ArgumentParser(description='COCO show')

    parser.add_argument('--image-dir', type=str,
                        default='/home/quan/Desktop/company/dirty_dataset/maskformer/images')
    parser.add_argument('--coco-json', type=str,
                        default='/home/quan/Desktop/company/dirty_dataset/maskformer/annotations/instances_default.json')
    parser.add_argument('--show-or-match', default=True)
    parser.add_argument('--type', type=str, help='custom|coco',
                        default='coco')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    coco_path = args.coco_json
    image_dir = args.image_dir

    coco_obj = coco.COCO(coco_path)
    num_category = len(coco_obj.cats)
    colors = (np.random.random((num_category+20, 3))*255)

    show_or_match = args.show_or_match
    idx_list = coco_obj.get_img_ids()
    if show_or_match:
        random.shuffle(idx_list)

    print('debug: count %d'%len(idx_list))
    for idx in idx_list:
        print('img info %d:' % idx)
        imgInfo = coco_obj.loadImgs(idx)[0]
        print(imgInfo)

        if show_or_match:
            image_path = os.path.join(image_dir, imgInfo['file_name'])
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            print('annotation info:')
            annIds = coco_obj.getAnnIds(imgIds=imgInfo['id'])
            annotations = coco_obj.loadAnns(annIds)
            print(annotations)

            if args.type=='custom':
                for anno in annotations:
                    cate_id = int(anno['category_id'])
                    x, y, w, h = anno['bbox']
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    cv2.rectangle(img, (x, y), (x + w, y + h), colors[cate_id], thickness=2)
                    cate_info = coco_obj.load_cats([cate_id])[0]
                    cate_name = cate_info['name']
                    cv2.putText(img, cate_name, (x+5, y+5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
                    # print(anno)

            plt.imshow(img)

            if args.type=='coco':
                coco_obj.showAnns(annotations, draw_bbox=False)

            plt.show()