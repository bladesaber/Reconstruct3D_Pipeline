import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import math
import imgaug.augmenters as iaa
import os
import random

def image_seamless_mixed(front_img, background_img, center_x, center_y):
    ### it is terrible

    # front_gray = cv2.cvtColor(front_img, cv2.COLOR_BGR2GRAY)
    # _, front_mask = cv2.threshold(front_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # front_img[front_mask==0, :] = 255

    mask = 255 * np.ones(front_img.shape[:2], np.uint8)
    # normal_clone = cv2.seamlessClone(front_img, background_img, mask, (320, 240), cv2.NORMAL_CLONE)
    mixed_clone = cv2.seamlessClone(front_img, background_img, mask, (center_x, center_y), cv2.MIXED_CLONE)
    # mono_clone = cv2.seamlessClone(front_img, background_img, mask, (320, 240), cv2.MONOCHROME_TRANSFER)

    return mixed_clone

def image_weight_mixed(front_img, background_img, center_x, center_y, weight):
    height, width, channel = background_img.shape
    xmin = int(center_x - width/2.0)
    ymin = int(center_y - height/2.0)
    layer_img = front_img.copy()
    layer_img[ymin:ymin+height, xmin:xmin+width, :] = background_img
    target_img = cv2.addWeighted(front_img, alpha=1.0-weight, src2=layer_img, beta=weight, gamma=0.0)
    return target_img

def get_color_hist(img, mask, pos_thresold=255):
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, mask = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    select_data = img[mask == pos_thresold, :]
    select_df = pd.DataFrame(select_data // 5.0 * 5.0, columns=['b', 'g', 'r'])
    select_df = select_df.value_counts()
    select_rgb = np.array(list(select_df.index))
    select_weight = select_df.values
    select_weight = select_weight / select_weight.sum()

    return select_rgb, select_weight

def rgb_hist_sample(img, mask, rgb_array, weights, pos_thresold=255, add_noise=True):
    num = (mask==pos_thresold).sum()

    range_idx = np.arange(0, weights.shape[0], 1)
    sample_idx = np.random.choice(range_idx, size=num, p=weights)
    sample_data = rgb_array[sample_idx, :]

    if add_noise:
        noise = np.random.uniform(-10.0, 10.0, size=sample_data.shape)
        sample_data = (sample_data + noise).astype(np.uint8)

    img[mask==pos_thresold, :] = sample_data

    return img

def polyline_smoothing(points, num_points=300):
    '''
    :param points: [X, Y]

    Test1:
        x = np.array([-0.25, -0.625, -0.125, -1.25, -1.125, -1.25, 0.875, 1.0, 1.0, 0.5, 1.0, 0.625, -0.25])
        y = np.array([1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 1.875, 1.75, 1.625, 1.5, 1.375, 1.25, 1.25])
        input_points = np.concatenate((x.reshape((-1, 1)), y.reshape((-1, 1))), axis=1)
        polyline_smoothing(points=input_points)

    Test2:
        x = np.array([1, 5, 5, 1])
        y = np.array([1, 1, 5, 5])
        input_points = np.concatenate((x.reshape((-1, 1)), y.reshape((-1, 1))), axis=1)
        polyline_smoothing(points=input_points, width=20, height=20)

    '''
    if points.shape[0]==3:
        interp_points = points
    elif points.shape[0]==4:
        interp_points = np.array([
            [points[:, 0].min(), points[:, 1].min()],
            [points[:, 0].min(), points[:, 1].max()],
            [points[:, 0].max(), points[:, 1].max()],
            [points[:, 0].max(), points[:, 1].min()]
        ])
    else:
        orig_len = points.shape[0]
        points = np.concatenate((points[-3:-1, :], points), axis=0)
        points = np.concatenate((points, points[1:3, :]), axis=0)

        t = np.arange(points.shape[0])
        fun_x = interp1d(t, points[:, 0], kind='cubic')
        fun_y = interp1d(t, points[:, 1], kind='cubic')

        interp_ti = np.linspace(2, orig_len + 1, num_points)
        interp_x = fun_x(interp_ti)
        interp_y = fun_y(interp_ti)

        interp_points = np.concatenate(
            (interp_x.reshape((-1, 1)), interp_y.reshape((-1, 1))),
            axis=1
        )

    # fig, ax = plt.subplots()
    # ax.plot(points[:, 0], points[:, 1])
    # ax.plot(interp_points[:, 0], interp_points[:, 1])
    # ax.margins(0.05)
    # plt.show()

    interp_points = interp_points.astype(np.int32)

    return interp_points

def points_to_mask(points, width, height):
    mask = np.zeros((height, width), dtype=np.uint8)

    ### only polt line
    # cv2.polylines(mask, pts=[interp_points], isClosed=True, color=255, thickness=1)

    # cv2.fillPoly(mask, pts=[interp_points], color=255)
    cv2.fillConvexPoly(mask, points=points, color=255)

    # cv2.imshow('d', mask)
    # cv2.waitKey(0)

    return mask

def random_polygon(num_points, min_length, max_length):
    angles = np.random.uniform(1e-4, 1.0, size=num_points)
    angles = angles/angles.sum()
    angles = math.pi * 2.0 * angles
    angles = np.cumsum(angles)

    lengths = np.random.uniform(min_length, max_length, size=num_points)

    points = []
    for idx, angle in enumerate(angles[:-1]):
        length = lengths[idx]
        x_dif = length * math.cos(angle)
        y_dif = length * math.sin(angle)
        points.append([x_dif, y_dif])
    points = np.array(points)
    return points

def random_elastic(image, alpha=0.5):
    elastic_parser = iaa.ElasticTransformation(alpha=alpha, sigma=0.25)

    parser = iaa.Sequential([elastic_parser])

    if len(image.shape) == 3:
        image = image[np.newaxis, ...]
        image = parser(images=image)
        image = image[0, ...]
    else:
        image = parser(images=image)

    return image

def random_weather(image, method='rain'):
    if method=='rain':
        weather_parser = iaa.Rain(speed=(0.35, 0.65))
    elif method=='snow':
        weather_parser = iaa.Snowflakes(speed=(0.35, 0.65))
    elif method=='fog':
        weather_parser = iaa.Fog()
    elif method=='snow':
        weather_parser = iaa.imgcorruptlike.Snow(severity=1)
    elif method=='spatter':
        weather_parser = iaa.imgcorruptlike.Spatter(severity=1)
    else:
        raise ValueError

    parser = iaa.Sequential([weather_parser])

    if len(image.shape) == 3:
        image = image[np.newaxis, ...]
        image = parser(images=image)
        image = image[0, ...]
    else:
        image = parser(images=image)

    return image

def random_blur(image, ksize=5):
    image = cv2.blur(image, (ksize, ksize))
    return image

def draw_polygon(image, points, color=(255, 0, 0)):
    image = cv2.polylines(image, [points], isClosed=True, color=color, thickness=1)
    return image

def percentage_crop(image, percentage=0.1):
    if len(image.shape)==3:
        height, width, c = image.shape

        crop_width = int(width * (1. - percentage))
        crop_height = int(height * (1. - percentage))
        rest_width = width - crop_width - 1
        rest_height = height - crop_height - 1

        xmin = random.randint(0, rest_width)
        ymin = random.randint(0, rest_height)

        return image[ymin:ymin+crop_height, xmin:xmin+crop_width, :]
    else:
        height, width = image.shape

        crop_width = int(width * (1. - percentage))
        crop_height = int(height * (1. - percentage))
        rest_width = width - crop_width - 1
        rest_height = height - crop_height - 1

        xmin = random.randint(0, rest_width)
        ymin = random.randint(0, rest_height)

        return image[ymin:ymin + crop_height, xmin:xmin + crop_width]

def test_polygon_mask():
    ### too complex so I deprective

    texture_dir = '/home/quan/Desktop/company/dataset/texture'
    texture_paths = os.listdir(texture_dir)

    shape_dir = '/home/quan/Desktop/company/dataset/shape/mask'
    shape_paths = os.listdir(shape_dir)

    car_img = cv2.imread('/home/quan/Desktop/company/car_data/1_3.jpg')

    num_scar = 5

    src_height, src_width, src_channel = car_img.shape
    src_center_xs = np.random.uniform(low=0.15, high=0.85, size=num_scar)
    src_center_ys = np.random.uniform(low=0.15, high=0.85, size=num_scar)

    num_veters = np.random.randint(4, 12, size=num_scar)

    min_lengths = np.random.uniform(0.01, 0.05, size=num_scar)
    max_lengths = np.random.uniform(0.05, 0.1, size=num_scar)

    mask_paths = np.random.choice(shape_paths, size=num_scar)
    texture_paths = np.random.choice(texture_paths, size=num_scar)

    frontground = car_img.copy()
    for idx in range(num_scar):
        center_x = src_center_xs[idx]
        center_y = src_center_ys[idx]
        num_veter = num_veters[idx]
        min_length = min_lengths[idx]
        max_length = max_lengths[idx]

        mask_path = os.path.join(shape_dir, mask_paths[idx])
        texture_path = os.path.join(texture_dir, texture_paths[idx])

        dif_points = random_polygon(
            num_points=num_veter,
            min_length=min_length, max_length=max_length
        )
        dif_xmax = (np.abs(dif_points[:, 0])).max()
        dif_ymax = (np.abs(dif_points[:, 1])).max()

        ### -------------------
        shape_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        assert len(shape_img.shape) == 2
        texture_img = cv2.imread(texture_path, cv2.IMREAD_UNCHANGED)

        shape_height, shape_width = shape_img.shape
        shape_points = np.zeros(dif_points.shape)

        dif_x_scale = 0.5/dif_xmax
        dif_y_scale = 0.5/dif_ymax
        shape_points[:, 0] = (dif_points[:, 0] * dif_x_scale + 0.5) * shape_width
        shape_points[:, 1] = (dif_points[:, 1] * dif_y_scale + 0.5) * shape_height
        shape_points = shape_points.astype(np.int32)

        mask = points_to_mask(shape_points, width=shape_width, height=shape_height)
        mask = np.bitwise_and(mask, shape_img)
        ### if use this will cause scale uncorrect
        # ys, xs = np.where(mask>127)
        # mask = cv2.resize(mask[ys.min():ys.max(), xs.min():xs.max()], (shape_width, shape_height))

        texture = percentage_crop(texture_img, percentage=0.2)
        texture = cv2.resize(texture, (shape_img.shape[1], shape_img.shape[0]))
        ### if use cv2.seamlessClone
        # texture[mask<=127, :] = 255
        texture[mask <= 127, :] = 0

        in_width, in_height = int(shape_width/dif_x_scale)-1, int(shape_height/dif_y_scale)-1
        texture = cv2.resize(texture, (in_width, in_height))
        mask = cv2.resize(mask, (in_width, in_height))
        _, mask = cv2.threshold(mask, thresh=127, maxval=255, type=cv2.THRESH_BINARY)

        center_x = int(center_x * src_width)
        center_y = int(center_y * src_height)

        front_xmin = int(center_x - in_width/2.0)
        front_ymin = int(center_y - in_height/2.0)
        crop_orig = car_img[front_ymin:front_ymin+in_height, front_xmin:front_xmin+in_width, :]
        crop_orig[mask==255, :] = texture[mask==255, :]

        # ### deprective
        # car_img = image_seamless_mixed(
        #     front_img=texture, background_img=car_img,
        #     center_x=center_x, center_y=center_y
        # )

        # ### for debug
        # mask_img = np.tile(shape_img[..., np.newaxis], [1,1,3])
        # mask_img = draw_polygon(mask_img, shape_points)
        # plt.figure('mask_img')
        # plt.imshow(mask_img)
        # plt.figure('mask')
        # plt.imshow(mask)
        # plt.figure('texture')
        # plt.imshow(texture)
        # plt.figure('crop')
        # plt.imshow(crop_orig)
        # plt.figure('front')
        # plt.imshow(car_img)
        # plt.show()

        # ### for debug
        # points = np.zeros(dif_points.shape)
        # points[:, 0] = (dif_points[:, 0] + center_x) * src_width
        # points[:, 1] = (dif_points[:, 1] + center_y) * src_height
        # points = points.astype(np.int32)
        # car_img = draw_polygon(car_img, points)

    cv2.imshow('d', car_img)
    cv2.waitKey(0)

if __name__ == '__main__':
    # test_polygon_mask()

    image = cv2.imread('/home/quan/Desktop/company/dataset/c2dca001870105f8b82f14441aa96ce0.jpeg')
    texture_img = cv2.imread('/home/quan/Desktop/company/dataset/texture/11.jpg')
    texture_img = cv2.resize(texture_img, (300, 300))

    show_img = image_seamless_mixed(front_img=texture_img, background_img=image, center_x=1400, center_y=900)
    show_img = cv2.resize(show_img, (1280, 720))

    cv2.imshow('d', show_img)
    cv2.waitKey(0)