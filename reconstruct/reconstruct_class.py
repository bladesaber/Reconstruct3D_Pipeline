import numpy as np
import open3d
import copy
import pandas as pd
import matplotlib.pyplot as plt

from reconstruct.open3d_utils import create_img_from_numpy
from reconstruct.open3d_utils import create_intrinsics
from reconstruct.open3d_utils import create_rgbd_from_color_depth
from reconstruct.open3d_utils import create_pcd_from_rgbd
from reconstruct.open3d_utils import create_OdometryOption
from reconstruct.open3d_utils import create_scaleable_TSDF

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
    def __init__(self):
        self.trans_list = []
        self.tsdf_model = create_scaleable_TSDF(voxel_size=0.002, sdf_trunc=0.02)

    def init(self, color_img, depth_img, instrinc):
        self.instrinc = instrinc

        color_img = create_img_from_numpy(color_img)
        depth_img = create_img_from_numpy(depth_img)
        rgbd = create_rgbd_from_color_depth(
            color=color_img, depth=depth_img,
            depth_trunc=0.5, convert_rgb_to_intensity=False
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
                max_correspondence_distance=voxel_size*1.4
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
        color_img_raw = color_img.copy()
        color_img = create_img_from_numpy(color_img)
        depth_img = create_img_from_numpy(depth_img)

        rgbd_current = create_rgbd_from_color_depth(color=color_img, depth=depth_img,
                                                    depth_trunc=0.5,
                                                    convert_rgb_to_intensity=False)
        pcd_current = create_pcd_from_rgbd(rgbd_current, instrics=self.instrinc)

        trans_dif, information_matrix = self.multiscale_icp(
            source=self.pcd_prev, target=pcd_current,
            voxel_sizes=[0.01, 0.005],
            max_iters=[50, 30],
            icp_method='point_to_plane'
        )
        trans_current = np.dot(trans_dif, trans_current)

        self.tsdf_model.integrate(rgbd_current, intrinsic=self.instrinc, extrinsic=trans_current)
        self.pcd_prev = pcd_current

        return trans_current, trans_dif, color_img_raw

    def draw_registration_result_original_color(self, source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.transform(transformation)

        source_temp = source_temp.voxel_down_sample(0.005)
        target_temp = target_temp.voxel_down_sample(0.005)

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
    def __init__(self):
        self.tsdf_model = create_scaleable_TSDF(voxel_size=0.002, sdf_trunc=0.02)
        self.trans_list = []

    def init(self, color_img, depth_img, instrinc):
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
            depth_trunc=0.5, convert_rgb_to_intensity=False
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
                max_correspondence_distance=voxel_size*1.4
            )

            current_transformation = result_icp.transformation
            if idx == run_times - 1:
                information_matrix = open3d.pipelines.registration.get_information_matrix_from_point_clouds(
                    source_down, target_down, voxel_size * 1.4,
                    result_icp.transformation)

        print(result_icp)
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
        color_img = create_img_from_numpy(color_img)
        depth_img = create_img_from_numpy(depth_img)

        rgbd_current = create_rgbd_from_color_depth(color=color_img, depth=depth_img,
                                                    depth_trunc=0.5,
                                                    convert_rgb_to_intensity=False)
        pcd_current = create_pcd_from_rgbd(rgbd_current, instrics=self.instrinc)

        ### -----------------
        tsdf_pcd = self.tsdf_model.extract_point_cloud()
        tsdf_pcd.transform(trans_current)

        ### debug
        # open3d.visualization.draw_geometries([tsdf_pcd, pcd_current])

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

        width_limit = np.bitwise_and(uvd[:, 0]<639, uvd[:, 0]>0)
        height_limit = np.bitwise_and(uvd[:, 1]<479, uvd[:, 1]>0)
        select_idx = np.bitwise_and(width_limit, height_limit)
        uvd = uvd[select_idx]
        tsdf_pcd_color = tsdf_pcd_color[select_idx]

        uvd_rgb = np.concatenate((uvd, tsdf_pcd_color), axis=1)
        uvd_rgb_df = pd.DataFrame(uvd_rgb, columns=['u', 'v', 'd', 'r', 'g', 'b'])
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

        pcd_ref = open3d.geometry.PointCloud()
        pcd_ref.points = open3d.utility.Vector3dVector(uvd_np)
        pcd_ref.colors = open3d.utility.Vector3dVector(rgb_np)

        ###------ debug
        # open3d.visualization.draw_geometries([pcd_ref])

        trans_dif, information_matrix = self.multiscale_icp(
            source=pcd_ref, target=pcd_current,
            voxel_sizes=[0.01],
            max_iters=[100],
            icp_method='point_to_plane',
        )
        trans_current = np.dot(trans_dif, trans_current)

        # tsdf_pcd.transform(trans_dif)
        # open3d.visualization.draw_geometries([
        #     tsdf_pcd,
        #     pcd_current
        # ])
        self.tsdf_model.integrate(rgbd_current, intrinsic=self.instrinc, extrinsic=trans_current)

        return trans_current, trans_dif, remap_img

class Visual_Odometry(object):
    def __init__(self):
        pass
