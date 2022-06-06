import open3d

from reconstruct.open3d_utils import create_img_from_numpy
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