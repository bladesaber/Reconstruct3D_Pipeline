import os
import cv2
import torch
import yaml
import random
import imgaug.augmenters as iaa
from torch.utils.data import Dataset
from typing import Dict
import numpy as np

from models.restormer.aug_utils import random_polygon
from models.restormer.aug_utils import points_to_mask

'''
2022-06-20
构造虚假数据集有较多的麻烦，伪造异常分布有很多的麻烦，目前的一些结论是:
（1）截图方法不是一个好方法
（2）抽样形成纹理虽然是一个马尔科夫的过程但实现难度以及限制条件太大
（3）我需要一些取巧，例如制定一些步骤
	1.限制整个分布到一个窄分布，尽管这个模型无法区分目标分布以及目标噪音，但这个模型能区分特定噪音（例如随机噪音）。
	2.问题是在图像特定修复时，现实中无法获得受污染的图片及其未受污染的图片，这就无法构成真实的图片对。所以图像降噪只能用于有限的降噪，对于无法模仿的噪声就无法特定的降噪。
	3.为了解决上面的问题，在获得特定噪音的图片后，由人使用可以模仿的噪音进行遮码，再经由1步骤的模型进行修复，修复图像再作为目标图像，进入训练体系。
	4.但这个方法是有一定的缺陷的，因为部分目标图像是虚假的，这就导致了误差是不可能从根本上收敛到目标的。
'''

class SimulateNoise_Parser(object):
    '''
    pre_image:
        JPEG_Compress, Blur, light_Blend, elastic_transform, rotate, translate, scale, crop
        contrast, sharpen
    noise_stage:
        polygon_random_color, polygon_pure_color, polygon_noise_color
        grip_black_corase, grip_color_corase, salt_pepper_black, salt_pepper_color
        salt_pepper_noise, guassian_noise
    '''

    def __init__(
            self,
            # shape_dir,
            # texture_dir
            config,
    ):
        # self.shape_dir = shape_dir
        # self.texture_dir = texture_dir
        # self.shape_paths = os.listdir(self.shape_dir)
        # self.texture_paths = os.listdir(self.texture_dir)

        self.config = config
        self.init_perImage_parser(config=self.config)
        self.init_noiseImage_parser(config=self.config)

    def init_perImage_parser(self, config:Dict):

        shuttle_p = config['color_shuttle']['p']
        self.channel_shuttle_parser = iaa.ChannelShuffle(p=shuttle_p)

        ### --------------------------------------------------
        self.group_rich_img = []
        if 'jpeg_compress' in config.keys():
            compression_min = config['jpeg_compress']['min']
            compression_max = config['jpeg_compress']['max']
            self.jpeg_parser = iaa.JpegCompression(compression=(compression_min, compression_max))
            self.group_rich_img.append(self.jpeg_parser)

        if 'guassian_blur' in config.keys():
            blur_min = config['guassian_blur']['min']
            blur_max = config['guassian_blur']['max']
            self.blur_guass_parser = iaa.GaussianBlur(sigma=(blur_min, blur_max))
            self.group_rich_img.append(self.blur_guass_parser)

        if 'light_Blend' in config.keys():
            nb_rows_min = config['light_Blend']['nb_rows_min']
            nb_rows_max = config['light_Blend']['nb_rows_max']
            nb_cols_min = config['light_Blend']['nb_cols_min']
            nb_cols_max = config['light_Blend']['nb_cols_max']
            self.blend_parser = iaa.BlendAlphaRegularGrid(
                nb_rows=(nb_rows_min, nb_rows_max),
                nb_cols=(nb_cols_min, nb_cols_max),
                background=iaa.Multiply(0.7),
                # alpha=[0.3, 0.7]
            )
            self.group_rich_img.append(self.blend_parser)

        # if 'elastic_transform' in config.keys():
        #     alpha_min = config['elastic_transform']['min']
        #     alpha_max = config['elastic_transform']['max']
        #     self.elastic_parser = iaa.ElasticTransformation(alpha=(alpha_min, alpha_max), sigma=0.25)

        ### --------------------------------------------------
        self.group_geometry = []
        if 'geometry' in config.keys():
            rotate_min = config['geometry']['rotate_min']
            rotate_max = config['geometry']['rotate_max']

            translate_p = config['geometry']['translate']

            scale_min = config['geometry']['scale_min']
            scale_max = config['geometry']['scale_max']

            self.geometry_parser = iaa.Affine(
                rotate=(rotate_min, rotate_max),
                translate_percent={"x": (-translate_p, translate_p), "y": (-translate_p, translate_p)},
                scale={"x": (scale_min, scale_max), "y": (scale_min, scale_max)},
            )
            self.group_geometry.append(self.geometry_parser)

        if 'crop' in config.keys():
            crop_p = config['crop']['p']
            self.crop_parser = iaa.CropAndPad(percent=(-crop_p, crop_p))
            self.group_geometry.append(self.crop_parser)

        ### -------------------------------------------------
        self.group_condition_adapt = []
        if 'contrast' in config.keys():
            gamma_min = config['contrast']['gamma_min']
            gamma_max = config['contrast']['gamma_max']
            self.contrast_parser = iaa.GammaContrast((gamma_min, gamma_max))
            self.group_condition_adapt.append(self.contrast_parser)

        if 'sharpen' in config.keys():
            alpha_min = config['sharpen']['alpha_min']
            alpha_max = config['sharpen']['alpha_max']
            lightness_min = config['sharpen']['lightness_min']
            lightness_max = config['sharpen']['lightness_max']
            self.sharpen_parser = iaa.Sharpen(alpha=(alpha_min, alpha_max), lightness=(lightness_min, lightness_max))
            self.group_condition_adapt.append(self.sharpen_parser)

    def init_noiseImage_parser(self, config:Dict):
        self.noise_group = []

        if 'corase_black' in config.keys():
            pmin = config['corase_black']['pmin']
            pmax = config['corase_black']['pmax']
            self.corase_black_parser = iaa.CoarseDropout(p=(pmin, pmax), size_percent=(0.3, 0.6))
            self.noise_group.append(self.corase_black_parser)

        if 'corase_color' in config.keys():
            pmin = config['corase_color']['pmin']
            pmax = config['corase_color']['pmax']
            self.corase_color_parser = iaa.CoarseDropout(p=(pmin, pmax), size_percent=(0.3, 0.6), per_channel=True)
            self.noise_group.append(self.corase_color_parser)

        if 'cutout_constant' in config.keys():
            n_iter_min = config['cutout_constant']['n_iter_min']
            n_iter_max = config['cutout_constant']['n_iter_max']
            size_min = config['cutout_constant']['size_min']
            size_max = config['cutout_constant']['size_max']
            self.cutout_constant_parser = iaa.Cutout(
                nb_iterations=(n_iter_min, n_iter_max),
                size=(size_min, size_max),
                fill_mode="constant",
                cval=(0, 255),
                fill_per_channel=0.5
            )
            self.noise_group.append(self.cutout_constant_parser)

        if 'saltAndpepper_black' in config.keys():
            pmin = config['saltAndpepper_black']['pmin']
            pmax = config['saltAndpepper_black']['pmax']
            self.saltAndpepper_black_parser = iaa.CoarseSaltAndPepper((pmin, pmax), size_percent=(0.3, 0.6))
            self.noise_group.append(self.saltAndpepper_black_parser)

        if 'saltAndpepper_color' in config.keys():
            pmin = config['saltAndpepper_color']['pmin']
            pmax = config['saltAndpepper_color']['pmax']
            self.saltAndpepper_color_parser = iaa.CoarseSaltAndPepper((pmin, pmax), size_percent=(0.3, 0.6), per_channel=True)
            self.noise_group.append(self.saltAndpepper_color_parser)

        if 'cutout_guass' in config.keys():
            n_iter_min = config['cutout_guass']['n_iter_min']
            n_iter_max = config['cutout_guass']['n_iter_max']
            size_min = config['cutout_guass']['size_min']
            size_max = config['cutout_guass']['size_max']
            self.cutout_guass_parser = iaa.Cutout(
                nb_iterations=(n_iter_min, n_iter_max),
                size=(size_min, size_max),
                fill_mode="gaussian",
                fill_per_channel=True
            )
            self.noise_group.append(self.cutout_guass_parser)

    def aug_image(self, image):
        ndim = image.ndim
        assert ndim in [3, 4]
        if ndim == 3:
            image = image[np.newaxis, ...]
            change_dim = True
        else:
            change_dim = False

        perImg_parser_list = []

        if random.uniform(0.0, 1.0) > 0.5:
            perImg_parser_list.append(self.channel_shuttle_parser)

        if random.uniform(0.0, 1.0) > 0.5:
            rich_parser = random.choice(self.group_rich_img)
            perImg_parser_list.append(rich_parser)

        if random.uniform(0.0, 1.0) > 0.5:
            geometry_parser = random.choice(self.group_geometry)
            perImg_parser_list.append(geometry_parser)

        if random.uniform(0.0, 1.0) > 0.5:
            condition_parser = random.choice(self.group_condition_adapt)
            perImg_parser_list.append(condition_parser)

        noise_parser_list = []
        # if random.uniform(0.0, 1.0) > 0.5:
        #     # sample_num = random.randint(1, 2)
        #     # noise_parser = np.random.choice(self.noise_group, size=sample_num, replace=False)
        #     # noise_parser_list.extend(noise_parser)
        #     noise_parser = random.choice(self.noise_group)
        #     noise_parser_list.append(noise_parser)
        noise_parser = random.choice(self.noise_group)
        noise_parser_list.append(noise_parser)

        # ### debug
        # for i in perImg_parser_list:
        #     print(i)

        if len(perImg_parser_list)>0:
            perImg_parser = iaa.Sequential(perImg_parser_list, random_order=True)
            image = perImg_parser(images=image)

        noise_image = image.copy()
        if len(noise_parser_list)>0:
            noise_parser = iaa.Sequential(noise_parser_list, random_order=True)
            noise_image = noise_parser(images=noise_image.copy())

        if change_dim:
            image = image[0, ...]
            noise_image = noise_image[0, ...]

        return image, noise_image

    def polygon_randomNoise(self, image, method='random_color'):
        height, width, c = image.shape

        num_veter = random.randint(4, 12)
        min_length = random.uniform(0.01, 0.03)
        max_length = random.uniform(0.03, 0.06)

        center_x = random.uniform(max_length, 1.0 - max_length)
        center_y = random.uniform(max_length, 1.0 - max_length)

        dif_points = random_polygon(num_points=num_veter, min_length=min_length, max_length=max_length)
        points = np.zeros(dif_points.shape)
        points[:, 0] = (dif_points[:, 0] + center_x) * width
        points[:, 1] = (dif_points[:, 1] + center_y) * height
        points = points.astype(np.int32)

        mask = points_to_mask(points, width=width, height=height)
        num_valid_points = (mask == 255).sum()

        if method=='random_color':
            image[mask == 255, :] = (np.random.uniform(0, 255, size=(num_valid_points, 3))).astype(np.uint8)
        elif method=='pure_color':
            color = np.random.uniform(0, 255, size=(1, 3))
            image[mask == 255, :] = (np.tile(color, [num_valid_points, 1])).astype(np.uint8)
        else:
            raise ValueError

        return image

    def shape_randomTexture(self, image):
        '''
        由于shape图片的尺寸无法与车的数据集图片形成合理的对应，导致形状特征可能在resize阶段完全消失，这就无意义了。如果不做resize，
        截取的形状受原shape图影响，截取图像可能无意义。因此丢弃，采取多边形代替。
        '''
        height, width, c = image.shape

        shape_path = random.choice(self.shape_paths)
        shape_img = cv2.imread(os.path.join(self.shape_dir, shape_path), cv2.IMREAD_UNCHANGED)
        texture_path = random.choice(self.texture_paths)
        texture_img = cv2.imread(os.path.join(self.texture_dir, texture_path), cv2.IMREAD_UNCHANGED)

        xlength = random.uniform(0.05, 0.1)
        ylength = random.uniform(0.05, 0.1)
        noise_xmin = random.uniform(xlength, 1.0 - xlength) - xlength / 2.0
        noise_ymin = random.uniform(ylength, 1.0 - ylength) - ylength / 2.

        xlength = int(xlength * width)
        ylength = int(ylength * height)
        noise_xmin = int(noise_xmin * width)
        noise_ymin = int(noise_ymin * height)

        shape_img = cv2.resize(shape_img, (xlength, ylength))
        texture_img = cv2.resize(texture_img, (xlength, ylength))

        crop_img = image[noise_ymin:noise_ymin + ylength, noise_xmin:noise_xmin + xlength, :]
        crop_img[shape_img > 127, :] = texture_img[shape_img > 127, :]
        image[noise_ymin:noise_ymin + ylength, noise_xmin:noise_xmin + xlength, :] = crop_img

        return image

    def polygon_randomTexture(self, image):
        height, width, c = image.shape

        num_veter = random.randint(4, 12)
        min_length = random.uniform(0.01, 0.03)
        max_length = random.uniform(0.03, 0.06)
        dif_points = random_polygon(num_points=num_veter, min_length=min_length, max_length=max_length)

        center_x = random.uniform(max_length, 1.0 - max_length)
        center_y = random.uniform(max_length, 1.0 - max_length)
        points = np.zeros(dif_points.shape)
        points[:, 0] = (dif_points[:, 0] + center_x) * width
        points[:, 1] = (dif_points[:, 1] + center_y) * height
        points = points.astype(np.int32)
        mask = points_to_mask(points, width=width, height=height)

        texture_path = random.choice(self.texture_paths)
        texture_img = cv2.imread(os.path.join(self.texture_dir, texture_path))
        texture_img = cv2.resize(texture_img, (width, height))

        texture_center_x = random.uniform(max_length, 1.0 - max_length)
        texture_center_y = random.uniform(max_length, 1.0 - max_length)
        texture_points = np.zeros(dif_points.shape)
        texture_points[:, 0] = (dif_points[:, 0] + texture_center_x) * width
        texture_points[:, 1] = (dif_points[:, 1] + texture_center_y) * height
        texture_points = points.astype(np.int32)
        texture_mask = points_to_mask(texture_points, width=width, height=height)

        # image_raw = draw_polygon(image.copy(), points)

        front_img = image.copy()
        front_img[mask == 255, :] = texture_img[texture_mask == 255, :]
        weight = random.uniform(0.0, 0.3)
        image = cv2.addWeighted(front_img, alpha=1.0 - weight, src2=image, beta=weight, gamma=0.0)

        return image

class Dataset_PairedImage(Dataset):

    def __init__(self, img_dir, config):
        super(Dataset_PairedImage, self).__init__()

        self.img_dir = img_dir
        self.img_paths = os.listdir(self.img_dir)

        self.config = config
        self.img_width = self.config['image_width']
        self.img_height = self.config['image_height']
        self.parser = SimulateNoise_Parser(config=self.config)

    def __getitem__(self, index):
        # index = index % len(self.img_paths)

        img_path = self.img_paths[index]
        img_name = img_path

        img = cv2.imread(os.path.join(self.img_dir, img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_width, self.img_height))

        # h, w, c = img.shape
        # assert c==3
        # assert h==self.img_height
        # assert w==self.img_width

        gt_image, noise_image = self.parser.aug_image(image=img)

        gt_image = gt_image.astype(np.float32)
        noise_image = noise_image.astype(np.float32)

        noise_image += np.random.random(noise_image.shape) * 5.0

        ### todo 需不需要归一化??
        gt_image = self.normalize(gt_image)
        noise_image = self.normalize(noise_image)

        gt_image = np.transpose(gt_image, (2, 0, 1))
        gt_image = torch.from_numpy(gt_image)

        noise_image = np.transpose(noise_image, (2, 0, 1))
        noise_image = torch.from_numpy(noise_image)

        return noise_image, gt_image, img_name
        # return {'noise': noise_image, 'gt':gt_image, 'name':img_name}

    def normalize(self, img):
        img_mean = np.mean(img, axis=(0, 1), keepdims=True)
        img_var = np.var(img, axis=(0, 1), keepdims=True)
        img_std = np.sqrt(img_var)
        return (img - img_mean)/img_std

    def __len__(self):
        return len(self.img_paths)

    def get_names(self):
        return self.img_paths

    def re_normalize(self, img):
        img_min = np.min(img, axis=(0, 1), keepdims=True)
        img_max = np.max(img, axis=(0, 1), keepdims=True)

        return ((img - img_min)/(img_max-img_min) * 255.).astype(np.uint8)

class RainDataset(Dataset):
    def __init__(self, img_dir):
        super(RainDataset, self).__init__()

if __name__ == '__main__':
    config_path = '/home/quan/Desktop/company/Reconstruct3D_Pipeline/models/restormer/test_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    dataset = Dataset_PairedImage(
        img_dir='/home/quan/Desktop/tempary/temp/good',
        config=config['dataset']
    )

    for batch in dataset:
        noise_image, gt_image, name = batch
        noise_image = noise_image.numpy()
        noise_image = np.transpose(noise_image, (1,2,0))
        noise_image = dataset.re_normalize(noise_image)
        noise_image = cv2.cvtColor(noise_image, cv2.COLOR_BGR2RGB)

        gt_image = gt_image.numpy()
        gt_image = np.transpose(gt_image, (1, 2, 0))
        gt_image = dataset.re_normalize(gt_image)
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)

        print(name)
        cv2.imshow('gt', gt_image)
        cv2.imshow('noise', noise_image)
        cv2.waitKey(0)

        break