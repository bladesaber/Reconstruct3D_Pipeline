import argparse
import os
import shutil
import numpy as np
import json

def parse_args():
    parser = argparse.ArgumentParser(description='COCO Merge')

    parser.add_argument('--input', type=str, nargs='+',
                        default=[
                            '/home/quan/Desktop/company/dirty_dataset/maskformer/annotations/1.json',
                            '/home/quan/Desktop/company/dirty_dataset/maskformer/annotations/3.json',
                        ])
    parser.add_argument('--output',
                        default='/home/quan/Desktop/company/dirty_dataset/maskformer/annotations/instances_default.json', type=str, help='')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    output_path = args.output
    files = args.input

    images_count = 1
    annotations_count = 1

    root_data = None
    for idx, file in enumerate(files):
        if idx == 0:
            with open(file, 'r', encoding='utf8') as f:
                raw_data = json.load(f)
            root_data = raw_data
            print('raw data:', len(raw_data['images']), len(raw_data['annotations']))

        else:
            with open(file, 'r', encoding='utf8') as f:
                raw_data = json.load(f)

            for cell in raw_data['images']:
                cell['id'] += images_count

            for cell in raw_data['annotations']:
                cell['image_id'] += images_count
                cell['id'] += annotations_count

            root_data['images'].extend(raw_data['images'])
            root_data['annotations'].extend(raw_data['annotations'])

            print('raw data:', len(raw_data['images']), len(raw_data['annotations']))

        images_count += len(raw_data['images'])
        annotations_count += len(raw_data['annotations'])

    print('raw data:', len(root_data['images']), len(root_data['annotations']))

    json_str = json.dumps(root_data, sort_keys=True, indent=4, separators=(',', ': '))
    with open(output_path, 'w') as f:
        f.write(json_str)