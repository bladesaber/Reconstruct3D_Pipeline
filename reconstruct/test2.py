import cv2
import open3d
import numpy as np

# pcd_open3d = open3d.geometry.PointCloud()
# pcd = np.random.random((100, 3))
# color = np.random.randint(0, 255, (100, 3))
# pcd_open3d.points = open3d.utility.Vector3dVector(pcd)
# pcd_open3d.points = open3d.utility.Vector3dVector(color)
#
# open3d.visualization.draw_geometries([pcd_open3d])

all_points = np.array([]).reshape((0, 3))
print(all_points.shape)