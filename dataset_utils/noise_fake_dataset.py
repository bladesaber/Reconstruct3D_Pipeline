import os
import cv2
import random
import numpy as np

from dataset_utils.aug_utils import random_polygon
from dataset_utils.aug_utils import percentage_crop
from dataset_utils.aug_utils import points_to_mask
from dataset_utils.aug_utils import draw_polygon

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

class SimulateNoise_Dataset(object):
    def __init__(self, shape_dir, texture_dir):
        self.shape_dir = shape_dir
        self.texture_dir = texture_dir
        self.shape_paths = os.listdir(self.shape_dir)
        self.texture_paths = os.listdir(self.texture_dir)

    def add_noise(self, image):
        pass

    def polygon_randomNoise(self, image):
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
        image[mask == 255, :] = (np.random.uniform(0, 255, size=(num_valid_points, 3))).astype(np.uint8)

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

if __name__ == '__main__':
    dataset = SimulateNoise_Dataset(
        shape_dir='/home/quan/Desktop/company/dataset/shape/mask',
        texture_dir='/home/quan/Desktop/company/dataset/texture'
    )

    car_img = cv2.imread('/home/quan/Desktop/company/car_data/1_3.jpg')
    # car_img = dataset.polygon_randomNoise(image=car_img)
    # car_img = dataset.shape_randomTexture(image=car_img)
    car_img = dataset.polygon_randomTexture(image=car_img)

    cv2.imshow('d', car_img)
    cv2.waitKey(0)
