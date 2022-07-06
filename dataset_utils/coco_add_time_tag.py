import json
import argparse
import os
import datetime
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='COCO Merge')

    parser.add_argument('--coco', type=str,
                        default='/home/quan/Desktop/company/dirty_dataset/test_img/2.json')
    parser.add_argument('--img_dir', type=str,
                        default='/home/quan/Desktop/company/dirty_dataset/test_img/imgs')

    parser.add_argument('--out_dir', type=str,
                        default='/home/quan/Desktop/company/dirty_dataset/test_img/new')
    parser.add_argument('--out_coco', type=str,
                        default='/home/quan/Desktop/company/dirty_dataset/maskformer/annotations/3.json')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    file = args.coco
    with open(file, 'r', encoding='utf8') as f:
        raw_data = json.load(f)

    time_tag = (datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")

    for img_cell in raw_data['images']:
        file_name = img_cell['file_name']
        path = os.path.join(args.img_dir, file_name)

        if not os.path.exists(path):
            raise ValueError

        new_file_name = '%s_%s'%(time_tag, file_name)
        target_path = os.path.join(args.out_dir, new_file_name)
        img_cell['file_name'] = new_file_name
        shutil.copy(path, target_path)

    json_str = json.dumps(raw_data, sort_keys=True, indent=4, separators=(',', ': '))
    with open(args.out_coco, 'w') as f:
        f.write(json_str)
