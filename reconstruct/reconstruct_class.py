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

class RGBD_Odometry(object):
    '''
    First Step:
    model.init(
                color_img=color_img,
                depth_img=depth_img,
                instrinc=instrinc_open3d,
                trans_original=trans_odometry
            )
    Later Step:
    status, trans_odometry = model.intergret(
                color_img=color_img,
                depth_img=depth_img,
                trans_current=trans_odometry
            )
    Visualize Step:
    pcd = model.tsdf_model.extract_voxel_point_cloud()
    # pcd = model.tsdf_model.extract_point_cloud()
    open3d.visualization.draw_geometries([pcd])

    mesh = model.tsdf_model.extract_triangle_mesh()
    open3d.visualization.draw_geometries([mesh])
    '''

    def __init__(self):
        self.rgbd_prev = None
        self.instrinc = None
        self.trans_list = []
        self.tsdf_model = create_scaleable_TSDF()

    def init(self, color_img, depth_img, instrinc, trans_original):
        color_img = create_img_from_numpy(color_img)
        depth_img = create_img_from_numpy(depth_img)
        self.rgbd_prev = create_rgbd_from_color_depth(
            color=color_img, depth=depth_img,
            depth_trunc=1.0, convert_rgb_to_intensity=False
        )

        self.instrinc = instrinc
        self.trans_list.append(trans_original)
        self.tsdf_model.integrate(image=self.rgbd_prev,
                                  intrinsic=self.instrinc,
                                  extrinsic=trans_original)

    def intergret(self, color_img, depth_img, trans_current):
        color_img = create_img_from_numpy(color_img)
        depth_img = create_img_from_numpy(depth_img)

        rgbd_current = create_rgbd_from_color_depth(color=color_img, depth=depth_img,
                                                    depth_trunc=1.0, convert_rgb_to_intensity=False)

        option = create_OdometryOption(max_depth_diff=0.01, min_depth=0.02, max_depth=1.0)
        odo_init = np.array([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ])

        success, trans, info = open3d.pipelines.odometry.compute_rgbd_odometry(
            self.rgbd_prev, rgbd_current,
            self.instrinc, odo_init,
            open3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
            option
        )
        if success:
            self.rgbd_prev = rgbd_current
            trans_odometry = np.dot(trans, trans_current)
            self.trans_list.append(trans_odometry.copy())

            self.tsdf_model.integrate(image=rgbd_current, intrinsic=self.instrinc, extrinsic=trans_odometry)
            return True, trans_odometry
        else:
            return False, None

class ICP_Odometry(object):
    '''
    pcd_prev = model.init(
                color_img=color_img, depth_img=depth_img, instrinc=instrinc_open3d
            )
    trans_odometry, pcd_prev = model.compute(
                color_img=color_img, depth_img=depth_img,
                source_pcd=pcd_prev, trans_current=trans_odometry
            )
    model.trans_list.append(trans_odometry.copy())
    '''

    def __init__(self,
                 depth_trunc,
                 tsdf_voxel_size,
                 ):
        self.depth_trunc = depth_trunc
        self.tsdf_voxel_size = tsdf_voxel_size

        self.trans_list = []
        self.tsdf_model = create_scaleable_TSDF(
            voxel_size=self.tsdf_voxel_size,
            sdf_trunc=2 * self.tsdf_voxel_size
        )

    def init(self, color_img, depth_img, instrinc):
        self.instrinc = instrinc

        color_img = create_img_from_numpy(color_img)
        depth_img = create_img_from_numpy(depth_img)
        rgbd = create_rgbd_from_color_depth(
            color=color_img, depth=depth_img,
            depth_trunc=self.depth_trunc, convert_rgb_to_intensity=False
        )
        pcd = create_pcd_from_rgbd(rgbd=rgbd, instrics=self.instrinc)
        self.pcd_prev = pcd

        return pcd

    def multiscale_icp(
            self,
            source, target,
            voxel_sizes, max_iters,
            icp_method='point_to_plane',
            init_transformation=np.identity(4),
    ):
        current_transformation = init_transformation
        run_times = len(max_iters)

        for idx in range(run_times):  # multi-scale approach
            max_iter = max_iters[idx]
            voxel_size = voxel_sizes[idx]
            distance_threshold = voxel_size * 1.4

            source_down = source.voxel_down_sample(voxel_size)
            target_down = target.voxel_down_sample(voxel_size)

            result_icp = self.icp(
                source=source_down, target=target_down,
                max_iter=max_iter,
                distance_threshold=distance_threshold,
                icp_method=icp_method,
                init_transformation=current_transformation,
                radius=voxel_size * 2.0,
                max_correspondence_distance=voxel_size * 1.4
            )

            current_transformation = result_icp.transformation
            if idx == run_times - 1:
                information_matrix = open3d.pipelines.registration.get_information_matrix_from_point_clouds(
                    source_down, target_down, voxel_size * 1.4,
                    result_icp.transformation)

        # print(result_icp)
        # self.draw_registration_result_original_color(source, target, result_icp.transformation)

        return (result_icp.transformation, information_matrix)

    def icp(self,
            source, target,
            max_iter, distance_threshold,
            icp_method='color',
            init_transformation=np.identity(4),
            radius=0.02,
            max_correspondence_distance=0.01,
            ):
        if icp_method == "point_to_point":
            result_icp = open3d.pipelines.registration.registration_icp(
                source, target,
                distance_threshold,
                init_transformation,
                open3d.pipelines.registration.TransformationEstimationPointToPoint(),
                open3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))

        else:
            source.estimate_normals(
                open3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
            )
            target.estimate_normals(
                open3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
            )
            if icp_method == "point_to_plane":
                result_icp = open3d.pipelines.registration.registration_icp(
                    source, target,
                    distance_threshold,
                    init_transformation,
                    open3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    open3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
                )

            if icp_method == "color":
                # Colored ICP is sensitive to threshold.
                # Fallback to preset distance threshold that works better.
                # TODO: make it adjustable in the upgraded system.
                result_icp = open3d.pipelines.registration.registration_colored_icp(
                    source, target,
                    max_correspondence_distance,
                    init_transformation,
                    open3d.pipelines.registration.TransformationEstimationForColoredICP(),
                    open3d.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=1e-6,
                        relative_rmse=1e-6,
                        max_iteration=max_iter)
                )

            if icp_method == "generalized":
                result_icp = open3d.pipelines.registration.registration_generalized_icp(
                    source, target,
                    distance_threshold,
                    init_transformation,
                    open3d.pipelines.registration.
                        TransformationEstimationForGeneralizedICP(),
                    open3d.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=1e-6,
                        relative_rmse=1e-6,
                        max_iteration=max_iter)
                )

        ### debug
        # information_matrix = open3d.pipelines.registration.get_information_matrix_from_point_clouds(
        #     source, target, 0.01,
        #     result_icp.transformation
        # )
        # print(information_matrix)

        return result_icp

    def compute(self, color_img, depth_img, trans_current):
        start_preprocess = time.time()

        color_img_raw = color_img.copy()
        color_img = create_img_from_numpy(color_img)
        depth_img = create_img_from_numpy(depth_img)

        rgbd_current = create_rgbd_from_color_depth(color=color_img, depth=depth_img,
                                                    depth_trunc=self.depth_trunc,
                                                    convert_rgb_to_intensity=False)
        pcd_current = create_pcd_from_rgbd(rgbd_current, instrics=self.instrinc)

        cost_preprocess = time.time() - start_preprocess
        print('[DEBUG]: Cost of preprocess: ', cost_preprocess)

        start_icp = time.time()
        trans_dif, information_matrix = self.multiscale_icp(
            source=self.pcd_prev, target=pcd_current,
            voxel_sizes=[0.03],
            max_iters=[100],
            icp_method='point_to_plane'
        )
        cost_icp = time.time() - start_icp
        print('[DEBUG]: Cost of icp: ', cost_icp)

        trans_current = np.dot(trans_dif, trans_current)

        self.tsdf_model.integrate(rgbd_current, intrinsic=self.instrinc, extrinsic=trans_current)
        self.pcd_prev = pcd_current

        return True, trans_current, trans_dif, color_img_raw

    def draw_registration_result_original_color(self, source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.transform(transformation)

        source_temp = source_temp.voxel_down_sample(0.01)
        target_temp = target_temp.voxel_down_sample(0.01)

        source_pcd_np = np.asarray(source_temp.points)
        source_open3d = open3d.geometry.PointCloud()
        source_open3d.points = open3d.utility.Vector3dVector(source_pcd_np)
        source_open3d.colors = open3d.utility.Vector3dVector(
            np.tile(np.array([[0, 255, 0]], dtype=np.uint8), (source_pcd_np.shape[0], 1))
        )

        target_pcd_np = np.asarray(target_temp.points)
        target_open3d = open3d.geometry.PointCloud()
        target_open3d.points = open3d.utility.Vector3dVector(target_pcd_np)
        target_open3d.colors = open3d.utility.Vector3dVector(
            np.tile(np.array([[255, 0, 0]], dtype=np.uint8), (target_pcd_np.shape[0], 1))
        )

        open3d.visualization.draw_geometries([source_open3d, target_open3d])

class RayCasting_Odometry(object):
    '''
    model.init(
        color_img=color_img, depth_img=depth_img, instrinc=instrinc_open3d
    )
    trans_odometry, trans_dif, remap_img = model.compute(
                color_img=color_img, depth_img=depth_img,
                trans_current=trans_odometry
    )
    model.trans_list.append(trans_dif.copy())
    '''

    def __init__(self,
                 depth_trunc,
                 tsdf_voxel_size,
                 ):
        self.tsdf_model = create_scaleable_TSDF(
            voxel_size=tsdf_voxel_size,
            sdf_trunc=4 * tsdf_voxel_size
        )
        self.trans_list = []

        self.depth_trunc = depth_trunc
        self.tsdf_voxel_size = tsdf_voxel_size

    def init(self, color_img, depth_img, instrinc, ):
        self.instrinc = instrinc
        self.instrinc_np = np.array([
            [606.9685, 0, 326.858],
            [0, 606.1065, 244.878],
            [0, 0, 1]
        ])

        color_img = create_img_from_numpy(color_img)
        depth_img = create_img_from_numpy(depth_img)
        rgbd = create_rgbd_from_color_depth(
            color=color_img, depth=depth_img,
            depth_trunc=self.depth_trunc, convert_rgb_to_intensity=False
        )
        self.tsdf_model.integrate(rgbd, intrinsic=self.instrinc, extrinsic=np.identity(4))

    def multiscale_icp(
            self,
            source, target,
            voxel_sizes, max_iters,
            icp_method='point_to_plane',
            init_transformation=np.identity(4),
    ):
        current_transformation = init_transformation
        run_times = len(max_iters)

        for idx in range(run_times):  # multi-scale approach
            max_iter = max_iters[idx]
            voxel_size = voxel_sizes[idx]
            distance_threshold = voxel_size * 1.4

            source_down = source.voxel_down_sample(voxel_size)
            target_down = target.voxel_down_sample(voxel_size)

            result_icp = self.icp(
                source=source_down, target=target_down,
                max_iter=max_iter,
                distance_threshold=distance_threshold,
                icp_method=icp_method,
                init_transformation=current_transformation,
                radius=voxel_size * 2.0,
                max_correspondence_distance=voxel_size * 1.4
            )

            current_transformation = result_icp.transformation
            if idx == run_times - 1:
                information_matrix = open3d.pipelines.registration.get_information_matrix_from_point_clouds(
                    source_down, target_down, voxel_size * 1.4,
                    result_icp.transformation)

        # print(result_icp)
        # self.draw_registration_result_original_color(source, target, result_icp.transformation)

        return (result_icp.transformation, information_matrix)

    def icp(self,
            source, target,
            max_iter, distance_threshold,
            icp_method='color',
            init_transformation=np.identity(4),
            radius=0.02,
            max_correspondence_distance=0.01,
            ):
        if icp_method == "point_to_point":
            result_icp = open3d.pipelines.registration.registration_icp(
                source, target,
                distance_threshold,
                init_transformation,
                open3d.pipelines.registration.TransformationEstimationPointToPoint(),
                open3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))

        else:
            source.estimate_normals(
                open3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
            )
            target.estimate_normals(
                open3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
            )
            if icp_method == "point_to_plane":
                result_icp = open3d.pipelines.registration.registration_icp(
                    source, target,
                    distance_threshold,
                    init_transformation,
                    open3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    open3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
                )

            if icp_method == "color":
                # Colored ICP is sensitive to threshold.
                # Fallback to preset distance threshold that works better.
                # TODO: make it adjustable in the upgraded system.
                result_icp = open3d.pipelines.registration.registration_colored_icp(
                    source, target,
                    max_correspondence_distance,
                    init_transformation,
                    open3d.pipelines.registration.TransformationEstimationForColoredICP(),
                    open3d.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=1e-6,
                        relative_rmse=1e-6,
                        max_iteration=max_iter)
                )

            if icp_method == "generalized":
                result_icp = open3d.pipelines.registration.registration_generalized_icp(
                    source, target,
                    distance_threshold,
                    init_transformation,
                    open3d.pipelines.registration.
                        TransformationEstimationForGeneralizedICP(),
                    open3d.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=1e-6,
                        relative_rmse=1e-6,
                        max_iteration=max_iter)
                )

        ### debug
        # information_matrix = open3d.pipelines.registration.get_information_matrix_from_point_clouds(
        #     source, target, 0.01,
        #     result_icp.transformation
        # )
        # print(information_matrix)

        return result_icp

    def compute(self, color_img, depth_img, trans_current):
        start_preprocess = time.time()

        color_img = create_img_from_numpy(color_img)
        depth_img = create_img_from_numpy(depth_img)

        rgbd_current = create_rgbd_from_color_depth(color=color_img, depth=depth_img,
                                                    depth_trunc=self.depth_trunc,
                                                    convert_rgb_to_intensity=False)
        pcd_current = create_pcd_from_rgbd(rgbd_current, instrics=self.instrinc)

        cost_preprocess = time.time() - start_preprocess
        print('[DEBUG]: Cost of preprocess: ', cost_preprocess)

        ### -----------------
        tsdf_pcd = self.tsdf_model.extract_point_cloud()
        tsdf_pcd.transform(trans_current)

        ### debug
        # open3d.visualization.draw_geometries([tsdf_pcd, pcd_current])

        start_raycasting = time.time()
        tsdf_pcd_np = np.asarray(tsdf_pcd.points)
        tsdf_pcd_color = np.asarray(tsdf_pcd.colors)
        # inv_instrinc = np.linalg.inv(self.instrinc_np)
        uvd = (self.instrinc_np.dot(tsdf_pcd_np.T)).T
        uvd[:, 0] = uvd[:, 0] / uvd[:, 2]
        uvd[:, 1] = uvd[:, 1] / uvd[:, 2]

        # ### debug
        # ### you must set voxel size is enough small like 0.001 other than the image is too sparse
        # uv_color = (tsdf_pcd_color * 255.).astype(np.int)
        # x_int = uvd[:, 0].astype(np.int)
        # y_int = uvd[:, 1].astype(np.int)
        # remap_img = np.zeros((480, 640, 3))
        # remap_img[y_int, x_int, :] = uv_color
        # remap_img = remap_img.astype(np.uint8)
        # plt.imshow(remap_img)
        # plt.show()

        width_limit = np.bitwise_and(uvd[:, 0] < 639, uvd[:, 0] > 0)
        height_limit = np.bitwise_and(uvd[:, 1] < 479, uvd[:, 1] > 0)
        depth_limit = uvd[:, 2] > 0.1
        select_bool = np.bitwise_and(width_limit, height_limit)
        select_bool = np.bitwise_and(select_bool, depth_limit)
        uvd = uvd[select_bool]
        tsdf_pcd_color = tsdf_pcd_color[select_bool]

        uvd_rgb = np.concatenate((uvd, tsdf_pcd_color), axis=1)
        uvd_rgb_df = pd.DataFrame(uvd_rgb, columns=['u', 'v', 'd', 'r', 'g', 'b'])

        ### todo here need voxel
        uvd_rgb_df['u'] = uvd_rgb_df['u'].astype(np.int)
        uvd_rgb_df['v'] = uvd_rgb_df['v'].astype(np.int)
        uvd_rgb_df['d'] = (uvd_rgb_df['d'] // 0.01) * 0.01
        uvd_rgb_df = uvd_rgb_df[uvd_rgb_df['d'] < self.depth_trunc]
        uvd_rgb_df = uvd_rgb_df[uvd_rgb_df['d'] > 0.1]

        uvd_rgb_df.sort_values(by=['d'], inplace=True)
        uvd_rgb_df.drop_duplicates(keep='first', inplace=True)
        uvd_rgb_df_np = uvd_rgb_df.to_numpy()
        uvd_np = uvd_rgb_df_np[:, :3]
        rgb_np = uvd_rgb_df_np[:, 3:]

        # ###------ debug
        uv_color = (rgb_np * 255.).astype(np.int)
        x_int = uvd_np[:, 0].astype(np.int)
        y_int = uvd_np[:, 1].astype(np.int)
        remap_img = np.zeros((480, 640, 3))
        remap_img[y_int, x_int, :] = uv_color
        remap_img = remap_img.astype(np.uint8)
        # plt.imshow(remap_img)
        # plt.show()

        inv_instrinc = np.linalg.inv(self.instrinc_np)
        uvd_np[:, 0] = uvd_np[:, 0] * uvd_np[:, 2]
        uvd_np[:, 1] = uvd_np[:, 1] * uvd_np[:, 2]
        uvd_np = (inv_instrinc.dot(uvd_np.T)).T

        ### limit point cloud
        uvd_np = uvd_np[:30000, :]
        rgb_np = rgb_np[:30000, :]

        pcd_ref = open3d.geometry.PointCloud()
        pcd_ref.points = open3d.utility.Vector3dVector(uvd_np)
        pcd_ref.colors = open3d.utility.Vector3dVector(rgb_np)

        ###------ debug
        # open3d.visualization.draw_geometries([pcd_ref])

        cost_raycasting = time.time() - start_raycasting
        print('[DEBUG]: Cost of RayCasting: ', cost_raycasting)

        start_icp = time.time()
        trans_dif, information_matrix = self.multiscale_icp(
            source=pcd_ref, target=pcd_current,
            voxel_sizes=[0.03],
            max_iters=[100],
            icp_method='point_to_plane'
        )
        cost_icp = time.time() - start_icp
        print('[DEBUG]: Cost of icp: ', cost_icp)

        trans_current = np.dot(trans_dif, trans_current)

        # tsdf_pcd.transform(trans_dif)
        # open3d.visualization.draw_geometries([
        #     tsdf_pcd,
        #     pcd_current
        # ])
        self.tsdf_model.integrate(rgbd_current, intrinsic=self.instrinc, extrinsic=trans_current)

        return True, trans_current, trans_dif, remap_img

class PoseGraph_Odometry(object):
    pass

class Visual_Odometry(object):
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
