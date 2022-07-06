import os
import json
import cv2
import imgaug.augmenters as iaa
import numpy as np
import random
from typing import Dict, List
import datetime
import shutil
from tqdm import tqdm

class DualImage_Dataset(object):

    def init_sharpen_parser(self, alpha=0.5, lightness=1.0):
        parser = iaa.Sharpen(alpha=alpha, lightness=lightness)
        return parser

    def init_contrast_parser(self, gamma):
        parser = iaa.GammaContrast(gamma=gamma)
        return parser

    def init_translate_parser(self, x_px, y_px):
        parser = iaa.Affine(translate_px={'x': x_px, 'y': y_px})
        return parser

    def init_rotate_parser(self, angle):
        parser = iaa.Affine(rotate=angle)
        return parser

    def init_scale_parser(self, scale):
        parser = iaa.Affine(scale=scale)
        return parser

    def init_blur_parser(self, sigma):
        parser = iaa.GaussianBlur(sigma=sigma)
        return parser

    def add_noise(self, image, mean=0.0, std=1.0):
        noise = np.random.random(image.shape) * std + mean
        image = (image + noise).astype(image.dtype)
        return image

    def shuttle_color(self, image):
        channel_order = np.array([0, 1, 2])
        new_order = channel_order.copy()
        while True:
            np.random.shuffle(new_order)
            if (new_order==channel_order).sum()<3:
                break

        new_image = np.zeros(image.shape, dtype=image.dtype)
        for idx in range(3):
            order_idx = new_order[idx]
            new_image[:, :, idx] = image[:, :, order_idx]

        return new_image

    def add_abnormal_tag(self, image, method):
        h, w, c = image.shape

        tag_x_length = random.randint(int(0.1*w), int(w*0.4))
        tag_y_length = random.randint(int(0.1*h), int(0.4*h))

        left = random.randint(0, w-tag_x_length)
        top = random.randint(0, h-tag_y_length)
        right = left + tag_x_length
        buttom = top + tag_y_length

        mask = np.zeros((image.shape[0], image.shape[1]))
        mask[top:buttom, left:right] = 255
        abnormal_image = image.copy()

        num_abnormal = (mask==255).sum()
        if method == 'pure':
            pure_color = np.random.randint(1, 249, (1, 3))
            abnormal_image[mask==255, :] = np.tile(pure_color, [num_abnormal, 1])

        elif method == 'random':
            abnormal_image[mask==255, :] = np.random.randint(1, 249, (num_abnormal, 3))

        return abnormal_image, mask

    def create_dual_image(
            self,
            images:List[np.array], aug_dict:Dict,
            save_dir,
            train_num, validate_num,
    ):
        idx_list = range(len(images))

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        mask_dir = os.path.join(save_dir, 'mask')
        if not os.path.exists(mask_dir):
            os.mkdir(mask_dir)
        good_dir = os.path.join(save_dir, 'good')
        if not os.path.exists(good_dir):
            os.mkdir(good_dir)
        bad_dir = os.path.join(save_dir, 'bad')
        if not os.path.exists(bad_dir):
            os.mkdir(bad_dir)

        time_tag = (datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")
        record_dict = {}

        num = max(train_num, validate_num)
        for name_id in tqdm(range(num)):
            idx = random.choice(idx_list)
            image = images[idx]
            original_image = image.copy()
            name_jpg = '%d_%s.jpg' % (name_id, time_tag)

            record_dict[name_jpg] = {}

            parser_list = []

            if 'sharpen' in aug_dict.keys():
                alpha = random.choice(aug_dict['sharpen']['alpha'])
                lightness = random.choice(aug_dict['sharpen']['lightness'])
                parser = self.init_sharpen_parser(alpha=alpha, lightness=lightness)

                parser_list.append(parser)
                record_dict[name_jpg]['sharpen'] = {'alpha': alpha, 'lightness':lightness}

            if 'contrast' in aug_dict.keys():
                gamma = random.choice(aug_dict['contrast']['gamma'])
                parser = self.init_contrast_parser(gamma=gamma)

                parser_list.append(parser)
                record_dict[name_jpg]['contrast'] = {'gamma': gamma}

            if 'translate' in aug_dict.keys():
                x_px = random.choice(aug_dict['translate']['x_px'])
                y_px = random.choice(aug_dict['translate']['y_px'])
                parser = self.init_translate_parser(x_px=x_px, y_px=y_px)

                parser_list.append(parser)
                record_dict[name_jpg]['translate'] = {'x_px': x_px, 'y_px':y_px}

            if 'rotate' in aug_dict.keys():
                angle = random.choice(aug_dict['rotate']['angle'])
                parser = self.init_rotate_parser(angle=angle)

                parser_list.append(parser)
                record_dict[name_jpg]['rotate'] = {'angle': angle}

            if 'scale' in aug_dict.keys():
                scale = random.choice(aug_dict['scale']['scale'])
                parser = self.init_scale_parser(scale=scale)

                parser_list.append(parser)
                record_dict[name_jpg]['scale'] = {'scale': scale}

            if 'blur' in aug_dict.keys():
                sigma = random.choice(aug_dict['blur']['sigma'])
                parser = self.init_blur_parser(sigma=sigma)

                parser_list.append(parser)
                record_dict[name_jpg]['blur'] = {'sigma': sigma}

            if len(parser_list)>0:
                parser = iaa.Sequential(parser_list, random_order=True)

                image = image[np.newaxis, ...]
                image = (parser(images=image))[0, ...]

            if 'shuttle_color' in aug_dict.keys():
                image = self.shuttle_color(image)
                record_dict[name_jpg]['shuttle_color'] = True

            image = self.add_noise(image, mean=0.0, std=1.0)

            # if random.uniform(0.0, 1.0)>0.5:
            #     abnormal_image, mask = self.add_abnormal_tag(image, method='pure')
            #     record_dict[name_jpg]['tag_method'] = 'pure'
            # else:
            #     abnormal_image, mask = self.add_abnormal_tag(image, method='random')
            #     record_dict[name_jpg]['tag_method'] = 'random'
            abnormal_image, mask = self.add_abnormal_tag(image, method='random')
            record_dict[name_jpg]['tag_method'] = 'random'

            if name_id<train_num:
                cv2.imwrite(os.path.join(good_dir, name_jpg), original_image)

            if name_id<validate_num:
                cv2.imwrite(os.path.join(bad_dir, name_jpg), abnormal_image)
                cv2.imwrite(os.path.join(mask_dir, name_jpg), mask)

        output_file = os.path.join(save_dir, '%s.json'%time_tag)
        json_str = json.dumps(record_dict, sort_keys=True, indent=4, separators=(',', ': '))
        with open(output_file, 'w') as f:
            f.write(json_str)

def main():
    jpg_dir = '/home/quan/Desktop/company/dirty_dataset/test_img/good'
    jpg_paths = os.listdir(jpg_dir)
    images = []
    for path in tqdm(jpg_paths):
        img = cv2.imread(os.path.join(jpg_dir, path))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        images.append(img)

    dataset_creator = DualImage_Dataset()
    dataset_creator.create_dual_image(
        images=images, train_num=0, validate_num=50,
        # save_dir='/home/quan/Desktop/tempary/abnormal_dataset/blur',
        save_dir='/home/quan/Desktop/company/dirty_dataset/test_img',
        # save_dir='/home/quan/Desktop/tempary/abnormal_dataset/translate',
        # save_dir='/home/quan/Desktop/tempary/abnormal_dataset/rotate',
        # save_dir='/home/quan/Desktop/tempary/abnormal_dataset/complex2',
        aug_dict={
            # 'sharpen': {
            #     'alpha': [0.3, 0.5, 0.7, 0.9],
            #     'lightness': [0.75, 1.0, 1.25, 1.5]
            # },
            # 'contrast':{'gamma':[0.5, 1.5]},
            # 'shuttle_color': True,
            # 'translate':{
            #     'x_px': [-20, -10, -5, 5, 10, 20],
            #     'y_px': [-20, -10, -5, 5, 10, 20],
            # },
            # 'rotate': {'angle': [-10, -5, -3, 3, 5, 10]}
            # 'blur': {'sigma': [1.0, 3.0, 5.0]}
        }
    )

if __name__ == '__main__':
    # jpg_dir = '/home/quan/Desktop/tempary/MVTec/capsule/train/good'
    # save_dir = '/home/quan/Desktop/tempary/abnormal_dataset/JPEG_Good'
    # paths = os.listdir(jpg_dir)[: 50]
    # for path in paths:
    #     shutil.copy(
    #         os.path.join(jpg_dir, path), os.path.join(save_dir, path)
    #     )

    main()
