import numpy as np
import open3d
import cv2
import matplotlib.pyplot as plt
import time

from camera.camera_realsense import Camera_RealSense, Camera_Fake

from reconstruct.open3d_utils import create_img_from_numpy
from reconstruct.open3d_utils import create_intrinsics
from reconstruct.open3d_utils import create_rgbd_from_color_depth
from reconstruct.open3d_utils import create_pcd_from_rgbd
from reconstruct.open3d_utils import create_OdometryOption
from reconstruct.open3d_utils import create_scaleable_TSDF

from reconstruct.reconstruct_class import RGBD_Odometry
from reconstruct.reconstruct_class import ICP_Odometry

if __name__ == '__main__':
    # ## use for record color jpg and depth png
    # camera = Camera_RealSense(
    #     input_bag_file='/home/quan/Desktop/tempary/dataset/2022_05_30.bag',
    #     save_dir='/home/quan/Desktop/tempary/dataset'
    # )
    # instrincs = camera.get_instrics()
    # camera.save_instrincs(instrincs)
    # run_count = 0
    # while True:
    #     status, color_img, depth_img = camera.get_img()
    #
    #     if run_count>50:
    #         camera.save_to_color_depth(color_img=color_img, depth_img=depth_img, id=run_count)
    #     cv2.imshow('d', color_img)
    #     key = cv2.waitKey(1)
    #     if key==ord('q'):
    #         break
    #
    #     run_count += 1

    ### ----------------------------------------------------
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

    # model = RGBD_Odometry()
    model = ICP_Odometry()

    trans_odometry = np.identity(4)
    pcd_prev = None

    cost_time_avg = 0.0
    time_count = 0.0
    run_count = 0

    while True:
        status, color_img, depth_img = camera.get_img()

        if not status:
            break

        start_time = time.time()

        color_img = create_img_from_numpy(color_img)
        depth_img = create_img_from_numpy(depth_img)
        rgbd = create_rgbd_from_color_depth(
            color=color_img, depth=depth_img,
            depth_trunc=0.6, convert_rgb_to_intensity=False
        )

        ### debug for image show
        # plt.subplot(1, 2, 1)
        # plt.imshow(rgbd.color)
        # plt.subplot(1, 2, 2)
        # plt.imshow(rgbd.depth)
        # plt.show()

        ### debug for pcd show
        # pcd = create_pcd_from_rgbd(rgbd, instrics=instrinc_open3d)
        # pcd_down = pcd.voxel_down_sample(0.005)
        # open3d.visualization.draw_geometries([pcd_down])

        if run_count==0:
            pcd_prev = model.init(
                color_img=color_img, depth_img=depth_img, instrinc=instrinc_open3d
            )

        else:
            trans_odometry, pcd_prev = model.compute(
                color_img=color_img, depth_img=depth_img,
                source_pcd=pcd_prev, trans_current=trans_odometry
            )

        time_cost = time.time() - start_time
        cost_time_avg = (cost_time_avg * time_count + time_cost) / (time_count + 1.0)
        time_count += 1.0
        if run_count%10==0:
            print('[Debug]: fps estimate ', 1.0 / cost_time_avg)

        model.trans_list.append(trans_odometry.copy())

        run_count += 1
        if run_count == 10:
            break

    print(run_count)
    np.save('/home/quan/Desktop/tempary/dataset/icp.npy', model.trans_list)

