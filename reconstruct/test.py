import threading

import numpy as np
import open3d
import cv2
import matplotlib.pyplot as plt
import time
from threading import Thread

from camera.camera_realsense import Camera_RealSense
from camera.fake_camera import Camera_Fake_2

from reconstruct.open3d_utils import create_img_from_numpy
from reconstruct.open3d_utils import create_intrinsics
from reconstruct.open3d_utils import create_rgbd_from_color_depth
from reconstruct.open3d_utils import create_pcd_from_rgbd
from reconstruct.open3d_utils import create_OdometryOption
from reconstruct.open3d_utils import create_scaleable_TSDF

from reconstruct.performance_utils import NonBlock_Visualizer

from reconstruct.reconstruct_class import RGBD_Odometry
from reconstruct.reconstruct_class import ICP_Odometry
from reconstruct.reconstruct_class import RayCasting_Odometry

def main():
    camera = Camera_Fake_2(
        save_dir='/home/quan/Desktop/template/redwood-3dscan/data/rgbd/00003',
    )

    camera_instrics = camera.load_instrincs(
        intrinsics_path='/home/quan/Desktop/work/Reconstruct3D_Pipeline/dataset/instrincs.json'
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

    # model = RGBD_Odometry()
    model = ICP_Odometry(
        depth_trunc=3.0, tsdf_voxel_size=0.01
    )
    # model = RayCasting_Odometry(
    #     depth_trunc=3.0, tsdf_voxel_size=0.01
    # )

    cv_sleep_time = 0
    run_count = 0
    trans_odometry = np.identity(4)

    # ### ---------------------------------------------------------
    # # vis = NonBlock_Visualizer(without_nvidia=False)
    # vis = open3d.visualization.Visualizer()
    # vis.create_window()
    # vis_pcd = open3d.geometry.PointCloud()
    # colors = np.tile(np.array([[255, 0, 0]], dtype=np.uint8), (1000, 1))
    # points = np.random.random((1000, 3))
    # vis_pcd.points = open3d.utility.Vector3dVector(points)
    # vis_pcd.colors = open3d.utility.Vector3dVector(colors)
    # vis.add_geometry(vis_pcd)
    # ### ---------------------------------------------------------

    while True:

        status, color_img, depth_img = camera.get_img()

        if not status:
            break

        # # ### debug for image show
        # color_img = create_img_from_numpy(color_img)
        # depth_img = create_img_from_numpy(depth_img)
        # rgbd = create_rgbd_from_color_depth(
        #     color=color_img, depth=depth_img,
        #     depth_trunc=3.0, convert_rgb_to_intensity=False
        # )
        # # plt.subplot(1, 2, 1)
        # # plt.imshow(rgbd.color)
        # # plt.subplot(1, 2, 2)
        # # plt.imshow(rgbd.depth)
        # # plt.show()
        #
        # ### debug for pcd show
        # pcd = create_pcd_from_rgbd(rgbd, instrics=instrinc_open3d)
        # pcd_down = pcd.voxel_down_sample(0.005)
        # open3d.visualization.draw_geometries([pcd_down])

        if run_count==0:
            model.init(
                color_img=color_img, depth_img=depth_img, instrinc=instrinc_open3d
            )

        else:
            trans_odometry, trans_dif, remap_img = model.compute(
                color_img=color_img, depth_img=depth_img,
                trans_current=trans_odometry
            )
            model.trans_list.append(trans_dif.copy())

        # tsdf_pcd = model.tsdf_model.extract_point_cloud()
        # vis_pcd.points = tsdf_pcd.points
        # vis_pcd.colors = tsdf_pcd.colors
        # vis.update_geometry(vis_pcd)
        # vis.poll_events()
        # vis.update_renderer()

        if run_count > 0:
            cv2.imshow('remap', remap_img)
            key = cv2.waitKey(cv_sleep_time)
            if key == ord('q'):
                break
            elif key == ord('p'):
                cv_sleep_time = 0
            elif key == ord('o'):
                cv_sleep_time = 1
            elif key == ord('v'):
                tsdf_pcd = model.tsdf_model.extract_point_cloud()
                open3d.visualization.draw_geometries([tsdf_pcd])
            else:
                pass

        run_count += 1
        # if run_count == 10:
        #     break

    # # vis.threat.join()
    tsdf_pcd = model.tsdf_model.extract_point_cloud()
    open3d.visualization.draw_geometries([tsdf_pcd])

if __name__ == '__main__':
    main()