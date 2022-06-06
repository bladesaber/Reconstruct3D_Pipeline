import numpy as np
import open3d
import copy
import pandas as pd
import matplotlib.pyplot as plt
import time
import cv2

from reconstruct.open3d_utils import create_img_from_numpy
from reconstruct.open3d_utils import create_rgbd_from_color_depth
from reconstruct.open3d_utils import create_pcd_from_rgbd
from reconstruct.open3d_utils import create_OdometryOption
from reconstruct.open3d_utils import create_scaleable_TSDF

from reconstruct.utils import rmsd_kabsch

class Visual_RGBD_Odometry(object):
    def __init__(self,
                 depth_trunc,
                 tsdf_voxel_size,
                 ):
        self.tsdf_model = create_scaleable_TSDF(
            voxel_size=tsdf_voxel_size,
            sdf_trunc=2 * tsdf_voxel_size
        )
        self.trans_list = []

        self.depth_trunc = depth_trunc
        self.tsdf_voxel_size = tsdf_voxel_size

    def init(self, color_img, depth_img, instrinc,):
        self.instrinc = instrinc
        self.instrinc_np = np.array([
            [606.9685, 0, 326.858],
            [0, 606.1065, 244.878],
            [0, 0, 1]
        ])

        self.prev_rgb_img = color_img.copy()
        self.prev_depth_img = depth_img.copy()

        color_img = create_img_from_numpy(color_img)
        depth_img = create_img_from_numpy(depth_img)
        rgbd = create_rgbd_from_color_depth(
            color=color_img, depth=depth_img,
            depth_trunc=self.depth_trunc, convert_rgb_to_intensity=False
        )
        self.tsdf_model.integrate(rgbd, intrinsic=self.instrinc, extrinsic=np.identity(4))

    def compute(self, color_img, depth_img, trans_current):
        status, trans_dif, remap_img = self.image_match(
            source_rgb_img=self.prev_rgb_img.copy(),
            target_rgb_img=color_img.copy(),
            source_depth_img=self.prev_depth_img.copy(),
            target_depth_img=depth_img.copy()
        )

        # ### debug
        # self.prev_rgb_img = color_img
        # self.prev_depth_img = depth_img
        # if status:
        #     h, w, _ = remap_img.shape
        #     remap_img = cv2.resize(remap_img, (int(w/3.0*2.0), int(h/3.0*2.0)))
        #     return True, None, None, remap_img
        # else:
        #     remap_img = color_img
        #     return False, None, None, remap_img
        # ### ------

        if status:
            trans_current = np.dot(trans_dif, trans_current)

            self.prev_rgb_img = color_img
            self.prev_depth_img = depth_img

            color_img = create_img_from_numpy(color_img)
            depth_img = create_img_from_numpy(depth_img)

            rgbd_current = create_rgbd_from_color_depth(color=color_img,
                                                        depth=depth_img,
                                                        depth_trunc=self.depth_trunc,
                                                        convert_rgb_to_intensity=False)
            self.tsdf_model.integrate(rgbd_current, intrinsic=self.instrinc, extrinsic=trans_current)

            h, w, _ = remap_img.shape
            remap_img = cv2.resize(remap_img, (int(w/2), int(h/2)))

            return True, trans_current, trans_dif, remap_img
        else:
            return False, trans_current, None, color_img

    def image_match(
            self,
            source_rgb_img,
            target_rgb_img,
            source_depth_img,
            target_depth_img,
    ):
        source_depth_img = source_depth_img/1000.0
        target_depth_img = target_depth_img/1000.0

        orb = cv2.ORB_create(
            # scaleFactor=1.2,
            # nlevels=8,
            # edgeThreshold=31,
            # firstLevel=0,
            # WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            # nfeatures=100,
            # patchSize=31

            nfeatures=500
        )
        [kp_s, des_s] = orb.detectAndCompute(source_rgb_img, None)
        [kp_t, des_t] = orb.detectAndCompute(target_rgb_img, None)
        if len(kp_s) == 0 or len(kp_t) == 0:
            return False, None, None

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des_s, des_t)

        pts_s = []
        pts_t = []
        for match in matches:
            target_x, target_y = kp_t[match.trainIdx].pt
            source_x, source_y = kp_s[match.queryIdx].pt

            source_depth = source_depth_img[int(source_y), int(source_x)]
            if source_depth > 3.0 or source_depth < 0.1:
                continue

            target_depth = target_depth_img[int(target_y), int(target_x)]
            if target_depth > 3.0 or target_depth < 0.1:
                continue

            pts_t.append(kp_t[match.trainIdx].pt)
            pts_s.append(kp_s[match.queryIdx].pt)

        pts_s = np.array(pts_s)
        pts_t = np.array(pts_t)

        # ### --- debug
        # draw_img = self.draw_correspondences(
        #     img_s=source_rgb_img.copy(), img_t=target_rgb_img.copy(),
        #     pts_s=pts_s.astype(np.int), pts_t=pts_t.astype(np.int)
        # )
        # plt.figure('raw orb')
        # plt.imshow(draw_img)
        # plt.show()
        # ### ------

        # focal_input = (self.instrinc_np[0, 0] + self.instrinc_np[1, 1]) / 2.0
        # pp_x = self.instrinc_np[0, 2]
        # pp_y = self.instrinc_np[1, 2]

        # Essential matrix is made for masking inliers
        # pts_s_int = np.int32(pts_s + 0.5)
        # pts_t_int = np.int32(pts_t + 0.5)
        # [E, mask] = cv2.findEssentialMat(pts_s_int,
        #                                  pts_t_int,
        #                                  focal=focal_input,
        #                                  pp=(pp_x, pp_y),
        #                                  method=cv2.RANSAC,
        #                                  prob=0.999,
        #                                  threshold=1.0)
        [E, mask] = cv2.findEssentialMat(
            pts_s,
            pts_t,
            cameraMatrix=self.instrinc_np,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )

        if mask is None:
            return False, None, None

        mask = (mask.reshape(-1)).astype(np.bool)
        pts_s = pts_s[mask]
        pts_t = pts_t[mask]

        # ### --- debug
        # draw_img = self.draw_correspondences(
        #     img_s=source_rgb_img.copy(), img_t=target_rgb_img.copy(),
        #     pts_s=pts_s.astype(np.int), pts_t=pts_t.astype(np.int)
        # )
        # # plt.figure('ransac orb')
        # # plt.imshow(draw_img)
        # # plt.show()
        # return True, None, draw_img
        # ### ------

        pts_s_int = pts_s.astype(np.int)
        pts_t_int = pts_t.astype(np.int)
        pts_s_depth = source_depth_img[pts_s_int[:, 1], pts_s_int[:, 0]]
        pts_t_depth = target_depth_img[pts_t_int[:, 1], pts_t_int[:, 0]]

        pts_xyz_s = np.concatenate((pts_s, pts_s_depth.reshape((-1, 1))), axis=1)
        pts_xyz_t = np.concatenate((pts_t, pts_t_depth.reshape((-1, 1))), axis=1)

        Kv = np.linalg.inv(self.instrinc_np)

        pts_xyz_s[:, 0] = pts_xyz_s[:, 0] * pts_xyz_s[:, 2]
        pts_xyz_s[:, 1] = pts_xyz_s[:, 1] * pts_xyz_s[:, 2]
        pts_xyz_s = pts_xyz_s.dot(Kv.T)

        pts_xyz_t[:, 0] = pts_xyz_t[:, 0] * pts_xyz_t[:, 2]
        pts_xyz_t[:, 1] = pts_xyz_t[:, 1] * pts_xyz_t[:, 2]
        pts_xyz_t = pts_xyz_t.dot(Kv.T)

        success, trans, mask = self.estimate_3D_transform_RANSAC(
            pts_xyz_s, pts_xyz_t
        )

        if success:
            pts_s = pts_s[mask]
            pts_t = pts_t[mask]

            ### --- debug
            # plt.figure('inlier ransac orb')
            draw_img = self.draw_correspondences(
                img_s=source_rgb_img.copy(), img_t=target_rgb_img.copy(),
                pts_s=pts_s.astype(np.int), pts_t=pts_t.astype(np.int)
            )
            # plt.imshow(draw_img)
            # plt.show()
            ### ------

            return True, trans, draw_img
        else:
            return False, None, None

    def draw_correspondences(
            self,
            img_s, img_t,
            pts_s, pts_t,
    ):
        height, width, channel = img_s.shape
        concat_img = np.zeros(shape=(height, width * 2, 3), dtype=np.uint8)
        concat_img[:height, :width, :] = img_s
        concat_img[:height, width:, :] = img_t

        for i in range(pts_s.shape[0]):
            sx = pts_s[i, 0]
            sy = pts_s[i, 1]
            tx = pts_t[i, 0] + width
            ty = pts_t[i, 1]

            cv2.line(
                concat_img,
                (sx, sy), (tx, ty),
                color=(255, 0, 0), thickness=1
            )
            cv2.circle(
                concat_img,
                center=(sx, sy), radius=3,
                color=(0, 255, 0), thickness=1
            )
            cv2.circle(
                concat_img,
                center=(tx, ty), radius=3,
                color=(0, 255, 0), thickness=1
            )

        return concat_img

    def estimate_3D_transform_RANSAC(self, pts_xyz_s, pts_xyz_t):
        max_iter = 1000
        max_distance = 0.03
        n_sample = 5
        n_points = pts_xyz_s.shape[0]
        Transform_good = np.identity(4)
        inlier_bool = None
        success = False

        # print('[DEBUG]" n_points ', n_points)

        if n_points < n_sample:
            return False, np.identity(4), []

        for i in range(max_iter):
            # todo sampling is a big problem
            rand_idx = np.random.randint(n_points, size=n_sample)
            sample_xyz_s = pts_xyz_s[rand_idx, :]
            sample_xyz_t = pts_xyz_t[rand_idx, :]

            R_approx, t_approx = self.estimate_3D_transform(
                src_points=sample_xyz_s,
                dst_points=sample_xyz_t
            )

            # evaluation
            diff_mat = pts_xyz_t - (pts_xyz_s.dot(R_approx) + t_approx)
            diff = np.linalg.norm(diff_mat, axis=1)
            inlier_bool = diff<max_distance
            n_inlier = inlier_bool.sum()

            # inlier_scale = n_inlier/float(n_points)
            # print('[DEBUG]" inlier scale ', inlier_scale)

            # note: diag(R_approx) > 0 prevents ankward transformation between
            # RGBD pair of relatively small amount of baseline.
            if (np.linalg.det(R_approx) != 0.0) and \
                (R_approx[0, 0] > 0 and R_approx[1, 1] > 0 and R_approx[2, 2] > 0) and \
                n_inlier>n_points * 0.8:
                Transform_good[:3, :3] = R_approx.T
                Transform_good[:3, 3] = [t_approx[0, 0], t_approx[0, 1], t_approx[0, 2]]

                inlier_scale = n_inlier/float(n_points)
                print('[DEBUG]" custom inlier scale ', inlier_scale)

                success = True
                break

        return success, Transform_good, inlier_bool

    def estimate_3D_transform(self, src_points, dst_points):
        src_center = np.mean(src_points, axis=0, keepdims=True)
        dst_center = np.mean(dst_points, axis=0, keepdims=True)

        src_center_normal = src_points - src_center
        dst_center_normal = dst_points - dst_center

        rot_mat = rmsd_kabsch.kabsch(
            P=src_center_normal, Q=dst_center_normal
        )
        tran_vec = dst_center - src_center.dot(rot_mat)

        return rot_mat, tran_vec