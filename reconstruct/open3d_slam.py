import os
import numpy as np
import open3d as o3d
import time

from camera.camera_realsense import Camera_RealSense, Camera_Fake
from reconstruct.open3d_utils import create_intrinsics
from reconstruct.open3d_utils import create_img_tensor_from_numpy

if __name__ == '__main__':
    camera = Camera_Fake(
        save_dir='/home/quan/Desktop/tempary/dataset',
        start_id=21
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
    instrinc_open3d = o3d.core.Tensor(instrinc_open3d.intrinsic_matrix, o3d.core.Dtype.Float64)

    device = o3d.core.Device('CPU:0')

    T_frame_to_model = o3d.core.Tensor(np.identity(4))
    model = o3d.t.pipelines.slam.Model(
        # voxel_size
        0.01,
        # block_resolution
        16,
        # block_count
        40000,
        # transformation
        T_frame_to_model,
        # device
        device
    )
    input_frame = o3d.t.pipelines.slam.Frame(640, 480, instrinc_open3d, device)
    raycast_frame = o3d.t.pipelines.slam.Frame(640, 480, instrinc_open3d, device)

    poses = []
    run_count = 0
    while True:
        start = time.time()

        status, color_img, depth_img = camera.get_img()
        if not status:
            break

        color = create_img_tensor_from_numpy(color_img)
        depth = create_img_tensor_from_numpy(depth_img)
        color = color.to(device)
        depth = depth.to(device)

        input_frame.set_data_from_image('depth', depth)
        input_frame.set_data_from_image('color', color)

        if run_count > 0:
            result = model.track_frame_to_model(input_frame, raycast_frame,
                                                depth_scale=1000.0,
                                                depth_max=0.5,
                                                depth_diff=0.01)
            T_frame_to_model = T_frame_to_model @ result.transformation

        poses.append(T_frame_to_model.cpu().numpy())
        model.update_frame_pose(run_count, T_frame_to_model)
        model.integrate(input_frame, depth_scale=1000.0, depth_max=0.5, trunc_voxel_multiplier=8.0)
        model.synthesize_model_frame(raycast_frame,
                                     depth_scale=1000.0,
                                     depth_min=0.0,
                                     depth_max=0.5,
                                     trunc_voxel_multiplier=8.0,
                                     enable_color=False
                                     )
        stop = time.time()
        print('slam takes {:.4}s'.format(stop - start))

        run_count += 1

    volume = model.voxel_grid
    mesh = volume.extract_triangle_mesh(weight_threshold=3.0)
    mesh = mesh.to_legacy()
    # o3d.visualization.draw([mesh])
    o3d.io.write_triangle_mesh('/home/quan/Desktop/tempary/1.ply', mesh)