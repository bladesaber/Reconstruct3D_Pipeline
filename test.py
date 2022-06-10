import cv2
import numpy as np

class Calibration_PoseFinder(object):
    def __init__(self):
        self.K = np.identity(3)
        self.Kv = np.identity(3)

        self.points = None

    def detect_harris(self, gray, opt=1, score=200):
        # Detector parameters
        blockSize = 2
        apertureSize = 3
        k = 0.04

        dst = cv2.cornerHarris(gray, blockSize, apertureSize, k)

        dst_norm = np.empty(dst.shape, dtype=np.float32)
        cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        corners_y, corners_x = np.where(dst_norm>score)
        corners_y = (np.array(corners_y)).reshape((-1, 1))
        corners_x = (np.array(corners_x)).reshape((-1, 1))
        corners = np.concatenate([corners_x, corners_y], axis=1)

        corners = corners[:, np.newaxis, :].astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        refine_corners = []

        corners = corners.reshape((-1, 2))
        corners_int = corners//1.0
        corners_unique = np.unique(corners_int, axis=0)
        for corner_label in corners_unique:
            group_bool = (np.sum(corners_int==corner_label, axis=1))==2.0
            group_corner = corners[group_bool]
            refine_corners.append(np.mean(group_corner, axis=0))

        refine_corners = np.array(refine_corners)

        print('[DEBUG]: The Corner Shape: ',refine_corners.shape)

        return refine_corners

    def compute_pose_orig(self, corner, dst_points, K):
        ### todo 我的精度不够
        ### opencv 的 src to dst 与我的相反
        homography, _ = cv2.findHomography(dst_points, corner, method=0)

        Kv = np.linalg.inv(K)
        r1 = Kv.dot(homography[:, 0:1])
        r2 = Kv.dot(homography[:, 1:2])

        r1_length = np.sqrt(np.sum(np.power(r1, 2)))
        r2_length = np.sqrt(np.sum(np.power(r2, 2)))

        s = np.sqrt((r1_length * r2_length))

        t = Kv.dot(homography[:, 2])
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

    def draw_pose_axes(self, rgb, length, RT):
        '''
        蓝色z，红色x，绿色y
        :param K:
            [fx, 0, cx]
            [0, fy, cy]
            [0, 0,  1]
        :return:

        '''
        rvec, _ = cv2.Rodrigues(RT[:3, :3])
        tvec = RT[:3, 3]

        dcoeffs = np.zeros(5)

        opoints = np.float32([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]).reshape(-1, 3) * length

        ipoints, _ = cv2.projectPoints(opoints, rvec, tvec, self.K, dcoeffs)
        ipoints = np.round(ipoints).astype(int)

        center = tuple(ipoints[0].ravel())

        cv2.line(rgb, center, tuple(ipoints[1].ravel()), (0, 0, 255), 2)
        cv2.line(rgb, center, tuple(ipoints[2].ravel()), (0, 255, 0), 2)
        cv2.line(rgb, center, tuple(ipoints[3].ravel()), (255, 0, 0), 2)

    def draw_points(self, img, points_2d):
        for point in points_2d:
            x = int(point[0])
            y = int(point[1])
            cv2.circle(img, (x, y), 2, (0, 255, 0), 2)
        return img

    def start_opencv_debug(self, img, points):
        self.img_raw = img
        self.show_img = img.copy()
        self.points = points

        cv2.namedWindow('debug')
        cv2.setMouseCallback('debug', model.opencv_event)

        while True:
            cv2.imshow('debug', self.show_img)

            key = cv2.waitKey(10)
            if key == ord('q'):
                break

    def opencv_event(self, event, x, y, flags, param):
        # print('[DEBUG]: Mouse X: ', x)
        # print('[DEBUG]: Mouse Y: ', y)

        if self.points is not None:
            current_pos = np.array([[x, y]])

            distance_square = np.sum(np.power(self.points - current_pos, 2), axis=1)
            min_index = np.argmin(distance_square)

            self.show_img = self.img_raw.copy()
            point_x = int(self.points[min_index][0])
            point_y = int(self.points[min_index][1])
            cv2.circle(self.show_img, (point_x, point_y), 2, (0, 255, 0), 2)

if __name__ == '__main__':
    img = cv2.imread('/home/quan/Desktop/company/Reconstruct3D_Pipeline/location/chessboard.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    model = Calibration_PoseFinder()
    corners = model.detect_harris(gray=gray, score=120)

    model.start_opencv_debug(img=img, points=corners)
