import open3d
import numpy as np
import cv2
import matplotlib.pyplot as plt

def create_img_from_numpy(
        img:np.array
)->open3d.geometry.Image:
    return open3d.geometry.Image(img)

def create_img_tensor_from_numpy(
        img:np.array
)->open3d.geometry.Image:
    return open3d.t.geometry.Image(img)

def create_rgbd_from_color_depth(
    color: open3d.geometry.Image,
    depth: open3d.geometry.Image,
    depth_scale=1000.0,
    depth_trunc=3.0,
    convert_rgb_to_intensity=True,
)->open3d.geometry.RGBDImage:
    ### if convert_rgb_to_intensity is True, can not use TSDFVolumeColorType
    return open3d.geometry.RGBDImage.create_from_color_and_depth(
        color=color, depth=depth,
        depth_scale=depth_scale, depth_trunc=depth_trunc,
        convert_rgb_to_intensity=convert_rgb_to_intensity
    )

def create_pcd_from_rgbd(
        rgbd: open3d.geometry.RGBDImage,
        instrics: open3d.camera.PinholeCameraIntrinsic
)->open3d.geometry.PointCloud:
    return open3d.geometry.PointCloud.create_from_rgbd_image(rgbd,instrics)

def create_intrinsics(width, height, fx, fy, cx, cy)->open3d.camera.PinholeCameraIntrinsic:
    instric = open3d.camera.PinholeCameraIntrinsic(
        width=width, height=height,
        fx=fx, fy=fy, cx=cx, cy=cy
    )
    return instric

def create_OdometryOption(
    iteration_number_per_pyramid_level=[20, 10, 5],
    max_depth_diff=0.05,
    min_depth = 0.0,
    max_depth = 4.000000
):
    option = open3d.pipelines.odometry.OdometryOption()
    # option.iteration_number_per_pyramid_level = iteration_number_per_pyramid_level
    option.max_depth_diff = max_depth_diff
    option.min_depth = min_depth
    option.max_depth = max_depth
    return option

def create_scaleable_TSDF(
    voxel_size=0.01,
    sdf_trunc=0.04,
    color_type=open3d.pipelines.integration.TSDFVolumeColorType.RGB8
):
    tsdf_model = open3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=color_type
    )
    return tsdf_model

def create_color_and_depth_path(color_path, depth_path):
    color_img = cv2.imread(color_path)
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    color_img = create_img_from_numpy(color_img)
    depth_img = create_img_from_numpy(depth_img)
    rgbd = create_rgbd_from_color_depth(
        color=color_img, depth=depth_img,
        depth_trunc=0.5, convert_rgb_to_intensity=False
    )
    return rgbd

if __name__ == '__main__':
    option = open3d.pipelines.odometry.OdometryOption()
    option.min_depth = 0.05
    print(option)
