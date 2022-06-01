import numpy as np
import open3d

from camera.camera_realsense import Camera_RealSense, Camera_Fake
from reconstruct.open3d_utils import create_intrinsics
from reconstruct.open3d_utils import create_scaleable_TSDF
from reconstruct.open3d_utils import create_img_from_numpy
from reconstruct.open3d_utils import create_rgbd_from_color_depth
from reconstruct.open3d_utils import create_pcd_from_rgbd

if __name__ == '__main__':
    trans_list = np.load('/home/quan/Desktop/tempary/dataset/icp.npy')
    trans_num = len(trans_list)

    camera = Camera_Fake(
        save_dir='/home/quan/Desktop/tempary/dataset',
        start_id=36
    )

    camera_instrics = camera.load_instrincs(
        intrinsics_path='/home/quan/Desktop/tempary/dataset/instrincs.json'
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

    all_points = np.array([]).reshape((0, 3))

    run_count = 0
    # tsdf = create_scaleable_TSDF(voxel_size=0.005, sdf_trunc=0.01)
    while True:
        status, color_img, depth_img = camera.get_img()
        color_img = create_img_from_numpy(color_img)
        depth_img = create_img_from_numpy(depth_img)
        rgbd = create_rgbd_from_color_depth(
            color=color_img, depth=depth_img, depth_trunc=0.5, convert_rgb_to_intensity=False
        )

        pcd = create_pcd_from_rgbd(rgbd=rgbd, instrics=instrinc_open3d)
        pcd_down = pcd.voxel_down_sample(voxel_size=0.005)
        pcd_np = np.asarray(pcd_down.points)

        pcd_open3d = open3d.geometry.PointCloud()
        pcd_open3d.points = open3d.utility.Vector3dVector(all_points)
        pcd_open3d.transform(trans_list[run_count])
        pcd_open3d.voxel_down_sample(0.005)
        all_points = np.asarray(pcd_open3d.points)

        all_points = np.concatenate((all_points, pcd_np), axis=0)

        # tsdf.integrate(
        #     image=rgbd, intrinsic=instrinc_open3d, extrinsic=trans_list[run_count]
        # )
        run_count += 1

        if run_count==trans_num-1:
            break

    ### -----------------------------------------
    # # pcd = tsdf.extract_voxel_point_cloud()
    # pcd = tsdf.extract_point_cloud()
    # open3d.visualization.draw_geometries([pcd])

    # mesh = tsdf.extract_triangle_mesh()
    # open3d.visualization.draw_geometries([mesh])

    pcd_open3d = open3d.geometry.PointCloud()
    pcd_open3d.points = open3d.utility.Vector3dVector(all_points)
    open3d.visualization.draw_geometries([pcd_open3d])
