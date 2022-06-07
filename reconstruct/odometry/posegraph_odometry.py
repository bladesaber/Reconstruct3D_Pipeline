import numpy as np
import open3d
import pandas as pd
import matplotlib.pyplot as plt
import time
import cv2
from queue import Queue
import os
import copy

from reconstruct.open3d_utils import create_img_from_numpy
from reconstruct.open3d_utils import create_rgbd_from_color_depth
from reconstruct.open3d_utils import create_pcd_from_rgbd
from reconstruct.open3d_utils import create_OdometryOption
from reconstruct.open3d_utils import create_scaleable_TSDF

class PoseGraph_Chunk_Odometry(object):

    def __init__(self,
                 depth_trunc,
                 tsdf_voxel_size,
                 ):
        self.tsdf_model = create_scaleable_TSDF(
            voxel_size=tsdf_voxel_size,
            sdf_trunc=3 * tsdf_voxel_size
        )
        self.trans_list = []

        self.depth_trunc = depth_trunc
        self.tsdf_voxel_size = tsdf_voxel_size

        self.queue_length = 10
        self.queue = Queue(maxsize=self.queue_length)
        self.rgbd_dict = {}

        self.template_id = 0

    def init(self, color_img, depth_img, instrinc,):
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
        pcd = create_pcd_from_rgbd(rgbd, instrics=self.instrinc)

        self.pcd_orig = pcd
        self.pcd_prev = pcd

        self.odometry_orig = np.identity(4)

        self.queue_id = 0
        self.queue.put(
            (
                self.queue_id, rgbd, None, None, None, None
            )
        )
        self.queue_id += 1

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

    def chunk_posegraph_opt(self):
        pose_graph = open3d.pipelines.registration.PoseGraph()
        trans_odometry = np.identity(4)

        id_prev = 0
        while not self.queue.empty():
            id_current, rgbd, trans_dif_from_prev, info_prev, trans_dif_from_orig, info_orig = self.queue.get()
            self.rgbd_dict[id_current] = rgbd

            if id_current==0:
                pose_graph.nodes.append(
                    open3d.pipelines.registration.PoseGraphNode(trans_odometry)
                )
                print('[DEBUG]: Add Node %d'%id_current)

            else:
                trans_odometry = trans_dif_from_prev.dot(trans_odometry)
                trans_odometry_inv = np.linalg.inv(trans_odometry)
                pose_graph.nodes.append(
                    open3d.pipelines.registration.PoseGraphNode(trans_odometry_inv)
                )
                print('[DEBUG]: Add Node %d' % id_current)

                pose_graph.edges.append(
                    open3d.pipelines.registration.PoseGraphEdge(
                        id_prev, id_current,
                        trans_dif_from_prev,
                        info_prev,
                        uncertain=False
                    )
                )
                print('[DEBUG]: Add Adjacent Edge from %d to %d' %(id_prev, id_current))

                ### loop closeure
                if trans_dif_from_orig is not None:
                    pose_graph.edges.append(
                        open3d.pipelines.registration.PoseGraphEdge(
                            0, id_current,
                            trans_dif_from_orig,
                            info_orig,
                            uncertain=True
                        )
                    )
                    print('[DEBUG]: Add Loop Closeure Edge from %d to %d' % (0, id_current))

            id_prev = id_current

        self.run_posegraph_optimization(pose_graph=pose_graph)

        return pose_graph

    def run_posegraph_optimization(
            self,
            pose_graph,
            max_correspondence_distance=0.05,
            preference_loop_closure=0.5
    ):

        # to display messages from o3d.pipelines.registration.global_optimization
        open3d.utility.set_verbosity_level(open3d.utility.VerbosityLevel.Debug)

        method = open3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt()
        criteria = open3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
        option = open3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=max_correspondence_distance,
            edge_prune_threshold=0.25,
            preference_loop_closure=preference_loop_closure,
            reference_node=0)

        open3d.pipelines.registration.global_optimization(pose_graph, method, criteria, option)

        open3d.utility.set_verbosity_level(open3d.utility.VerbosityLevel.Error)

    def raycasting_icp(self, pcd_current, trans_current):
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

        return trans_current, remap_img

    def compute(self, color_img, depth_img, trans_current):
        return_img = color_img.copy()

        color_img = create_img_from_numpy(color_img)
        depth_img = create_img_from_numpy(depth_img)

        rgbd_current = create_rgbd_from_color_depth(color=color_img, depth=depth_img,
                                                    depth_trunc=self.depth_trunc,
                                                    convert_rgb_to_intensity=False)
        pcd_current = create_pcd_from_rgbd(rgbd_current, instrics=self.instrinc)

        if self.queue.qsize()==1:
            trans_dif_from_prev, info_prev = self.multiscale_icp(
                source=self.pcd_prev, target=pcd_current,
                voxel_sizes=[0.03],
                max_iters=[100],
                icp_method='point_to_plane',
                init_transformation=np.identity(4)
            )

            self.queue.put(
                (
                    self.queue_id, rgbd_current,
                    trans_dif_from_prev, info_prev,
                    None, None
                )
            )

        else:
            trans_dif_from_prev, info_prev = self.multiscale_icp(
                source=self.pcd_prev, target=pcd_current,
                voxel_sizes=[0.03],
                max_iters=[100],
                icp_method='point_to_plane',
                init_transformation=np.identity(4)
            )

            trans_dif_from_orig, info_orig = self.multiscale_icp(
                source=self.pcd_orig, target=pcd_current,
                voxel_sizes=[0.03],
                max_iters=[100],
                icp_method='point_to_plane',
                init_transformation=np.identity(4)
            )

            self.queue.put(
                (
                    self.queue_id, rgbd_current,
                    trans_dif_from_prev, info_prev,
                    trans_dif_from_orig, info_orig
                )
            )

        if self.queue.full():
            # print('[DEBUG]: begin pose graph opt %d' % self.queue.qsize())

            pose_graph = self.chunk_posegraph_opt()
            node_num = len(pose_graph.nodes)

            # ### --- debug fragment visualization
            # local_tsdf_model = create_scaleable_TSDF(
            #     voxel_size=self.tsdf_voxel_size,
            #     sdf_trunc=3 * self.tsdf_voxel_size
            # )
            #
            # for i in range(node_num):
            #     rgbd = self.rgbd_dict[i]
            #
            #     pose = pose_graph.nodes[i].pose
            #     pose = np.linalg.inv(pose)
            #     local_tsdf_model.integrate(rgbd, self.instrinc, pose)
            #
            # # # open3d.visualization.draw_geometries([local_tsdf_model.extract_point_cloud()])
            # open3d.io.write_point_cloud(
            #     os.path.join('/home/quan/Desktop/company/template', '%d_local.ply'%self.template_id),
            #     local_tsdf_model.extract_point_cloud()
            # )
            # print('[DEBUG]: save template_id:%d'%self.template_id)
            # # self.template_id += 1
            # ### ------

            for i in range(node_num):
                rgbd = self.rgbd_dict[i]

                pose_dif = pose_graph.nodes[i].pose
                pose_dif = np.linalg.inv(pose_dif)
                pose = pose_dif.dot(self.odometry_orig)

                self.tsdf_model.integrate(rgbd, self.instrinc, pose)

            # ### ------ debug
            # open3d.io.write_point_cloud(
            #     os.path.join('/home/quan/Desktop/company/template', '%d_tsdf.ply'%self.template_id),
            #     self.tsdf_model.extract_point_cloud()
            # )
            # # self.template_id += 1
            # ### ---------

            self.odometry_orig = np.linalg.inv(pose_graph.nodes[node_num-1].pose).dot(self.odometry_orig)
            self.odometry_orig, remap_img = self.raycasting_icp(
                pcd_current=pcd_current,
                trans_current=self.odometry_orig
            )

            ### ------ debug
            template_pcd = open3d.geometry.PointCloud()

            template_pointcloud = np.asarray(pcd_current.points)
            template_color = np.tile(np.array([[255,0,0]]), [template_pointcloud.shape[0], 1])

            local_tsdf_pcd = open3d.geometry.PointCloud()
            local_tsdf_pcd.points = copy.deepcopy(self.tsdf_model.extract_point_cloud().points)
            local_tsdf_pcd.colors = copy.deepcopy(self.tsdf_model.extract_point_cloud().colors)
            local_tsdf_pcd.transform(self.odometry_orig)
            local_tsdf_pcd_pointcloud = np.asarray(local_tsdf_pcd.points)
            local_tsdf_pcd_colors = np.asarray(local_tsdf_pcd.colors)
            concat_pcd = np.concatenate(
                [template_pointcloud, local_tsdf_pcd_pointcloud], axis=0
            )
            concat_color = np.concatenate(
                [template_color, local_tsdf_pcd_colors], axis=0
            )

            template_pcd.points = open3d.utility.Vector3dVector(concat_pcd)
            template_pcd.colors = open3d.utility.Vector3dVector(concat_color)

            # open3d.visualization.draw_geometries([template_pcd])
            open3d.io.write_point_cloud(
                os.path.join('/home/quan/Desktop/company/template', '%d_compare.ply'%self.template_id),
                template_pcd
            )
            self.template_id +=1
            ### ------------

            ### --- init
            self.pcd_orig = pcd_current
            self.rgbd_dict.clear()
            self.queue_id = 0
            assert self.queue.qsize()==0

            self.queue.put(
                (
                    self.queue_id, rgbd_current, None, None, None, None
                )
            )

            return_img = remap_img

        self.pcd_prev = pcd_current
        self.queue_id += 1

        return True, None, None, return_img