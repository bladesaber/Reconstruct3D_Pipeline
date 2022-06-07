import numpy as np
import open3d
from queue import Queue
import os

from camera.fake_camera import Camera_Fake_2
from reconstruct.open3d_utils import create_intrinsics

from reconstruct.open3d_utils import create_img_from_numpy
from reconstruct.open3d_utils import create_rgbd_from_color_depth
from reconstruct.open3d_utils import create_pcd_from_rgbd
from reconstruct.open3d_utils import create_OdometryOption
from reconstruct.open3d_utils import create_scaleable_TSDF

def multiscale_icp(
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

        result_icp = icp(
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

    print(result_icp)
    # self.draw_registration_result_original_color(source, target, result_icp.transformation)

    return (result_icp.transformation, information_matrix)

def icp(
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
                open3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
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

def chunk_posegraph_opt(queue, rgbd_dict):
    pose_graph = open3d.pipelines.registration.PoseGraph()
    trans_odometry = np.identity(4)

    id_prev = 0
    while not queue.empty():
        id_current, rgbd, trans_dif_from_prev, info_prev, trans_dif_from_orig, info_orig = queue.get()
        rgbd_dict[id_current] = rgbd

        print('[DEBUG]: add %d to graph'%id_current)

        if id_current == 0:
            pose_graph.nodes.append(
                open3d.pipelines.registration.PoseGraphNode(trans_odometry)
            )

        else:
            trans_odometry = trans_dif_from_prev.dot(trans_odometry)
            trans_odometry_inv = np.linalg.inv(trans_odometry)
            pose_graph.nodes.append(
                open3d.pipelines.registration.PoseGraphNode(trans_odometry_inv)
            )

            pose_graph.edges.append(
                open3d.pipelines.registration.PoseGraphEdge(
                    id_prev, id_current,
                    trans_dif_from_prev,
                    info_prev,
                    uncertain=False
                )
            )

            if id_current > 1:
                pose_graph.edges.append(
                    open3d.pipelines.registration.PoseGraphEdge(
                        0, id_current,
                        trans_dif_from_orig,
                        info_orig,
                        uncertain=True
                    )
                )

        # id_prev = id_current

    run_posegraph_optimization(pose_graph=pose_graph)

    return pose_graph

def run_posegraph_optimization(
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


def test_posegraph():
    camera = Camera_Fake_2(
        save_dir='/home/quan/Desktop/company/Reconstruct3D_Pipeline/data/rgbd/00003',
    )

    camera_instrics = camera.load_instrincs(
        intrinsics_path='/home/quan/Desktop/company/Reconstruct3D_Pipeline/data/instrincs.json'
    )
    depth_instric = camera_instrics['depth']
    instrinc_open3d = create_intrinsics(
        width=depth_instric['width'],
        height=depth_instric['height'],
        fx=depth_instric['fx'],
        fy=depth_instric['fy'],
        cx=depth_instric['cx'],
        cy=depth_instric['cy']
    )

    img_pack, idx_list = camera.get_img_from_range(
        start_id=46, end_id=56
    )

    pcd_orig = None
    pcd_prev = None

    queue = Queue(maxsize=10)
    rgbd_dict = {}

    for count_id, idx in enumerate(idx_list):
        color_img = img_pack[idx]['color']
        depth_img = img_pack[idx]['depth']

        color_img = create_img_from_numpy(color_img)
        depth_img = create_img_from_numpy(depth_img)
        rgbd_current = create_rgbd_from_color_depth(color=color_img, depth=depth_img,
                                                    depth_trunc=3.0,
                                                    convert_rgb_to_intensity=False)
        pcd_current = create_pcd_from_rgbd(rgbd_current, instrics=instrinc_open3d)

        if count_id == 0:
            pcd_orig = pcd_current

            queue.put(
                (count_id, rgbd_current, None, None, None, None)
            )
            count_id += 1

        elif count_id == 1:
            trans_dif_from_prev, info_prev = multiscale_icp(
                source=pcd_prev, target=pcd_current,
                voxel_sizes=[0.03],
                max_iters=[100],
                icp_method='point_to_plane',
                init_transformation=np.identity(4)
            )
            queue.put(
                (
                    count_id, rgbd_current,
                    trans_dif_from_prev, info_prev,
                    None, None
                )
            )
            count_id += 1

        else:
            trans_dif_from_prev, info_prev = multiscale_icp(
                source=pcd_prev, target=pcd_current,
                voxel_sizes=[0.03],
                max_iters=[100],
                icp_method='point_to_plane',
                init_transformation=np.identity(4)
            )

            trans_dif_from_orig, info_orig = multiscale_icp(
                source=pcd_orig, target=pcd_current,
                voxel_sizes=[0.03],
                max_iters=[100],
                icp_method='point_to_plane',
                init_transformation=np.identity(4)
            )

            queue.put(
                (
                    count_id, rgbd_current,
                    trans_dif_from_prev, info_prev,
                    trans_dif_from_orig, info_orig
                )
            )
            count_id += 1

        pcd_prev = pcd_current

    print('-------------------------------------------------------------')
    pose_graph = chunk_posegraph_opt(queue=queue, rgbd_dict=rgbd_dict)
    node_num = len(pose_graph.nodes)

    local_tsdf_model = create_scaleable_TSDF(
        voxel_size=0.02,
        sdf_trunc=3 * 0.02
    )

    for i in range(node_num):
        rgbd = rgbd_dict[i]

        pose = pose_graph.nodes[i].pose
        pose = np.linalg.inv(pose)
        local_tsdf_model.integrate(rgbd, instrinc_open3d, pose)

    open3d.visualization.draw_geometries([local_tsdf_model.extract_point_cloud()])
    # open3d.io.write_point_cloud(
    #     os.path.join('/home/quan/Desktop/template/cache', '%d_local.ply' % 0),
    #     local_tsdf_model.extract_point_cloud()
    # )

if __name__ == '__main__':
    test_posegraph()
