import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def image_mixed(front_img, background_img, center_x, center_y):
    # front_gray = cv2.cvtColor(front_img, cv2.COLOR_BGR2GRAY)
    # _, front_mask = cv2.threshold(front_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # front_img[front_mask==0, :] = 255

    mask = 255 * np.ones(front_img.shape[:2], np.uint8)
    # normal_clone = cv2.seamlessClone(front_img, background_img, mask, (320, 240), cv2.NORMAL_CLONE)
    mixed_clone = cv2.seamlessClone(front_img, background_img, mask, (center_x, center_y), cv2.MIXED_CLONE)
    # mono_clone = cv2.seamlessClone(front_img, background_img, mask, (320, 240), cv2.MONOCHROME_TRANSFER)

    return mixed_clone

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

def polyline_smoothing(width, height, points, num_points=300):
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
    mask = np.zeros((height, width), dtype=np.uint8)

    ### only polt line
    # cv2.polylines(mask, pts=[interp_points], isClosed=True, color=255, thickness=1)

    # cv2.fillPoly(mask, pts=[interp_points], color=255)
    cv2.fillConvexPoly(mask, points=interp_points, color=255)

    # cv2.imshow('d', mask)
    # cv2.waitKey(0)

    return mask

if __name__ == '__main__':
    pass
