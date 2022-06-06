import numpy as np
import open3d
import time
import copy

from reconstruct.open3d_utils import create_img_from_numpy
from reconstruct.open3d_utils import create_rgbd_from_color_depth
from reconstruct.open3d_utils import create_pcd_from_rgbd
from reconstruct.open3d_utils import create_OdometryOption
from reconstruct.open3d_utils import create_scaleable_TSDF

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