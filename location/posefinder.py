import cv2
import numpy as np
import argparse
from queue import Queue

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ### left topic listener param
    parser.add_argument('--coodr_path', type=str,
                        default='/home/quan/Desktop/company/Reconstruct3D_Pipeline/location/coodr.txt')
    args, _ = parser.parse_known_args()
    return args

class Calibration_PoseFinder(object):

    def __init__(self, coodr_path):
        self.K = np.array([
            [606.96, 0, 326.85],
            [0, 606.10, 244.87],
            [0, 0, 1]
        ])
        self.Kv = np.linalg.inv(self.K)
        self.dcoeffs = np.zeros(5)

        self.points = np.array([])

        self.coodr_prefix = []
        with open(coodr_path, 'r') as f:
            content = f.readlines()
        for txt in content:
            x, y = txt.split(' ')
            x, y = float(x), float(y)
            self.coodr_prefix.append([x, y])
        self.coodr_prefix = np.array(self.coodr_prefix)

    def compute_axis(self, corner, dst_points):
        homography, _ = cv2.findHomography(dst_points, corner, method=0)

        r1 = self.Kv.dot(homography[:, 0:1])
        r2 = self.Kv.dot(homography[:, 1:2])

        r1_length = np.sqrt(np.sum(np.power(r1, 2)))
        r2_length = np.sqrt(np.sum(np.power(r2, 2)))

        s = np.sqrt((r1_length * r2_length))

        t = self.Kv.dot(homography[:, 2])
        if t[-1] < 0:
            s = s * -1

        t = t / s
        r1 = r1 / s
        r2 = r2 / s

        r3 = (np.cross(r1.T, r2.T)).reshape((-1, 1))
        rot = np.concatenate((r1, r2, r3), axis=1)

        U, S, V = np.linalg.svd(rot)
        rot = U.dot(V)

        RT = np.concatenate((rot, t.reshape((-1, 1))), axis=1)

        return RT

    ### ------ image process
    def canny(self, img, thresold_1=100, thresold_2=300):
        edge = cv2.Canny(img, thresold_1, thresold_2)
        return edge

    def sharpen(self, img, method='CUSTOM'):
        if method == 'CUSTOM':
            sharpen_op = np.array([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ], dtype=np.float32)
            sharpen_image = cv2.filter2D(img, cv2.CV_32F, sharpen_op)
            sharpen_image = cv2.convertScaleAbs(sharpen_image)
            return sharpen_image
        elif method == 'USM':
            blur_img = cv2.GaussianBlur(img, (0, 0), 5)
            usm = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)
            return usm

    def adaptiveThreshold(self, img, block_size=15):
        binary = cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
            block_size, 10
        )
        return binary

    def sobel(self, img, thresold=-1):
        x_grad = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        y_grad = cv2.Sobel(img, cv2.CV_32F, 0, 1)

        x_grad = cv2.convertScaleAbs(x_grad)
        y_grad = cv2.convertScaleAbs(y_grad)

        dst = cv2.add(x_grad, y_grad, dtype=cv2.CV_16S)
        dst = cv2.convertScaleAbs(dst)

        if thresold > 0:
            _, dst = cv2.threshold(dst, thresh=thresold, maxval=255, type=cv2.THRESH_BINARY)

        return dst

    def color_thresold(self, img, r_thresold, g_thresold, b_thresold, is_rgb):
        if is_rgb:
            r_bool = img[:, :, 0] > r_thresold
            g_bool = img[:, :, 1] > g_thresold
            b_bool = img[:, :, 2] > b_thresold
            mask = np.bitwise_and(r_bool, g_bool, b_bool)
        else:
            r_bool = img[:, :, 2] > r_thresold
            g_bool = img[:, :, 1] > g_thresold
            b_bool = img[:, :, 0] > b_thresold
            mask = np.bitwise_and(r_bool, g_bool, b_bool)

        return (mask * 255.).astype(np.uint8)

    def to_gray(self, img, is_rgb):
        if is_rgb:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray

    ### ------ corner detection
    def detect_harris(self, gray, score=200, blockSize=2, apertureSize=3):
        # Detector parameters
        blockSize = blockSize
        apertureSize = apertureSize
        k = 0.04

        dst = cv2.cornerHarris(gray, blockSize, apertureSize, k)
        harris_img = (dst - dst.min()) / (dst.max() - dst.min()) * 255.
        # cv2.imshow('d', harris_img.astype(np.uint8))
        # cv2.waitKey(0)

        harris_corners = harris_img > score

        corners_y, corners_x = np.where(harris_corners == 1)
        corners_y = (np.array(corners_y)).reshape((-1, 1))
        corners_x = (np.array(corners_x)).reshape((-1, 1))
        corners = np.concatenate([corners_x, corners_y], axis=1)

        corners = corners[:, np.newaxis, :].astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        refine_corners = []

        corners = corners.reshape((-1, 2))
        corners_int = corners // 1.0
        corners_unique = np.unique(corners_int, axis=0)
        for corner_label in corners_unique:
            group_bool = (np.sum(corners_int == corner_label, axis=1)) == 2.0
            group_corner = corners[group_bool]
            refine_corners.append(np.mean(group_corner, axis=0))

        refine_corners = np.array(refine_corners)

        # print('[DEBUG]: The Corner Shape: ', refine_corners.shape)

        return refine_corners

    ### ------ debug
    def draw_rect(self, rgb, RT):
        '''
        蓝色z，红色x，绿色y
        :param K:
            [fx, 0, cx]
            [0, fy, cy]
            [0, 0,  1]
        '''
        rvec, _ = cv2.Rodrigues(RT[:3, :3])
        tvec = RT[:3, 3]

        # opoints = np.array([
        #     ### center
        #     [0., 0., 0., 1.0],
        #
        #     ### corner
        #     [self.coodr_prefix[0, 0], self.coodr_prefix[0, 1], 0., 1.0],
        #     [self.coodr_prefix[1, 0], self.coodr_prefix[1, 1], 0., 1.0],
        #     [self.coodr_prefix[2, 0], self.coodr_prefix[2, 1], 0., 1.0],
        #     [self.coodr_prefix[3, 0], self.coodr_prefix[3, 1], 0., 1.0],
        #
        #     ### x, y, z
        #     [0.1, 0., 0., 1.0],
        #     [0., 0.1, 0., 1.0],
        #     [0., 0., 0.1, 1.0]
        #
        # ]).astype(np.float32)
        opoints = np.array([
            ### center
            [0., 0., 0.],

            ### corner
            [self.coodr_prefix[0, 0], self.coodr_prefix[0, 1], 0.],
            [self.coodr_prefix[1, 0], self.coodr_prefix[1, 1], 0.],
            [self.coodr_prefix[2, 0], self.coodr_prefix[2, 1], 0.],
            [self.coodr_prefix[3, 0], self.coodr_prefix[3, 1], 0.],

            ### x, y, z
            [0.1, 0., 0.],
            [0., 0.1, 0.],
            [0., 0., 0.1]

        ]).astype(np.float32)

        ipoints, _ = cv2.projectPoints(opoints, rvec, tvec, self.K, self.dcoeffs)
        # K = np.concatenate([self.K, np.zeros((3,1))], axis=1)
        # RT = np.concatenate((RT, np.array([[0., 0., 0., 1.]])), axis=0)
        # ipoints = (K.dot(RT.dot(opoints.T))).T
        # ipoints = ipoints[:, :2]/ipoints[:, -1:]

        ipoints = np.round(ipoints).astype(int)

        center = tuple(ipoints[0].ravel())

        ### draw line
        cv2.line(rgb, tuple(ipoints[1].ravel()), tuple(ipoints[2].ravel()), (255, 0, 255), 2)
        cv2.line(rgb, tuple(ipoints[2].ravel()), tuple(ipoints[3].ravel()), (255, 0, 255), 2)
        cv2.line(rgb, tuple(ipoints[3].ravel()), tuple(ipoints[4].ravel()), (255, 0, 255), 2)
        cv2.line(rgb, tuple(ipoints[4].ravel()), tuple(ipoints[1].ravel()), (255, 0, 255), 2)

        ### draw axis
        cv2.line(rgb, center, tuple(ipoints[5].ravel()), (0, 0, 255), 2)  # x
        cv2.line(rgb, center, tuple(ipoints[6].ravel()), (0, 255, 0), 2)  # y
        cv2.line(rgb, center, tuple(ipoints[7].ravel()), (255, 0, 0), 2)  # z

    def draw_points(self, img, points_2d):
        for point in points_2d:
            x = int(point[0])
            y = int(point[1])
            cv2.circle(img, (x, y), 3, (0, 255, 0), 1)
        return img

if __name__ == '__main__':
    args = parse_args()

    img = cv2.imread('/home/quan/Desktop/company/Reconstruct3D_Pipeline/location/test.jpg')
    # img = cv2.imread('/home/quan/Desktop/company/Reconstruct3D_Pipeline/location/chessboard.jpg')

    height, width, c = img.shape
    resize_width = 640
    resize_height = 480
    img = cv2.resize(img, (resize_width, resize_height))

    model = Calibration_PoseFinder(coodr_path=args.coodr_path)

    ### ----------------------------------------------------------------
    detection_corners = np.array([])
    mouse_x, mouse_y = 0.0, 0.0
    select_index = None
    RT = None
    coodr_queue = Queue(maxsize=model.coodr_prefix.shape[0])

    def opencv_event(event, x, y, flags, param):
        global mouse_x, mouse_y
        # print('[DEBUG]: Mouse X: ', x, ' Mouse Y:', y)
        mouse_x = x
        mouse_y = y

    cv2.namedWindow('debug')
    cv2.setMouseCallback('debug', opencv_event)

    while True:

        show_img = img.copy()

        gray = model.color_thresold(img=show_img, is_rgb=False, r_thresold=180, g_thresold=180, b_thresold=180)
        detection_corners = model.detect_harris(gray=gray, score=120, blockSize=3)

        # show_img = model.draw_points(img=show_img, points_2d=detection_corners)

        if detection_corners.shape[0] > 0:
            current_pos = np.array([[mouse_x, mouse_y]])

            distance_square = np.sum(np.power(detection_corners - current_pos, 2), axis=1)
            select_index = np.argmin(distance_square)

            point_x = int(detection_corners[select_index][0])
            point_y = int(detection_corners[select_index][1])
            cv2.circle(show_img, (point_x, point_y), 3, (0, 255, 0), 1)

        if RT is not None:
            model.draw_rect(rgb=show_img, RT=RT)

        cv2.imshow('debug', show_img)

        key = cv2.waitKey(5)
        if key == ord('q'):
            break

        elif key == ord('s'):
            coodr_queue.put(detection_corners[select_index])
            print('[DEBUG]: Adding Corner: ', detection_corners[select_index])

        elif key == ord('c'):

            uv_array = []
            while not coodr_queue.empty():
                uv_array.append(coodr_queue.get())
            uv_array = np.array(uv_array)

            print('-----------------')
            print(uv_array)
            print(model.coodr_prefix)

            assert uv_array.shape[0]==model.coodr_prefix.shape[0]
            RT = model.compute_axis(corner=uv_array, dst_points=model.coodr_prefix)
