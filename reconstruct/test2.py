# import time
#
# import cv2
# import open3d
# import numpy as np
# import matplotlib.pyplot as plt
#
# from reconstruct.open3d_utils import create_color_and_depth_path
# import pandas as pd
#
# # rgbd_1 = create_color_and_depth_path(
# #     color_path='/home/quan/Desktop/tempary/dataset/color/21.jpg',
# #     depth_path='/home/quan/Desktop/tempary/dataset/depth/21.png',
# # )
# # # rgbd_2 = create_color_and_depth_path(
# # #     color_path='/home/quan/Desktop/tempary/dataset/color/22.jpg',
# # #     depth_path='/home/quan/Desktop/tempary/dataset/depth/22.png',
# # # )
# #
# # plt.subplot(1, 2, 1)
# # plt.title('Redwood grayscale image')
# # plt.imshow(rgbd_1.color)
# # plt.subplot(1, 2, 2)
# # plt.title('Redwood depth image')
# # plt.imshow(rgbd_1.depth)
# # plt.show()
#
# import threading
#
# open3d.utility.set_verbosity_level(open3d.utility.VerbosityLevel.Debug)
#
# vis = open3d.visualization.Visualizer()
# vis.create_window()
#
# pcd = open3d.geometry.PointCloud()
# colors = np.tile(np.array([[255, 0, 0]], dtype=np.uint8), (1000, 1))
# points = np.random.random((1000, 3))
# pcd.points = open3d.utility.Vector3dVector(points)
# pcd.colors = open3d.utility.Vector3dVector(colors)
#
# # open3d.visualization.draw_geometries([pcd])
#
# vis.add_geometry(pcd)
#
# # for _ in range(100):
# #     pcd_current = open3d.geometry.PointCloud()
# #     colors = (np.random.randint(0, 255, (2000, 3))).astype(np.uint8)
# #     points = np.random.random((2000, 3))
# #     pcd_current.points = open3d.utility.Vector3dVector(points)
# #     pcd_current.colors = open3d.utility.Vector3dVector(colors)
# #
# #     pcd.points = pcd_current.points
# #     pcd.colors = pcd_current.colors
# #
# #     vis.update_geometry(pcd)
# #     vis.poll_events()
# #     vis.update_renderer()
# #
# #     time.sleep(1.0)
#
# def threat_run():
#     global vis, pcd
#
#     for _ in range(100):
#         pcd_current = open3d.geometry.PointCloud()
#         colors = (np.random.randint(0, 255, (2000, 3))).astype(np.uint8)
#         points = np.random.random((2000, 3))
#         pcd_current.points = open3d.utility.Vector3dVector(points)
#         pcd_current.colors = open3d.utility.Vector3dVector(colors)
#
#         pcd.points = pcd_current.points
#         pcd.colors = pcd_current.colors
#
#         vis.update_geometry(pcd)
#         vis.poll_events()
#         vis.update_renderer()
#
#         time.sleep(1.0)
#
# t = threading.Thread(target=threat_run)
# t.start()
# print('asd')
# t.join()
#
# # def run():
# #     while True:
# #         pcd_current = open3d.geometry.PointCloud()
# #         colors = (np.random.randint(0, 255, (2000, 3))).astype(np.uint8)
# #         points = np.random.random((2000, 3))
# #         pcd_current.points = open3d.utility.Vector3dVector(points)
# #         pcd_current.colors = open3d.utility.Vector3dVector(colors)
# #
# #         pcd.points = pcd_current.points
# #         pcd.colors = pcd_current.colors
# #
# #         vis.update_geometry(pcd)
# #         vis.poll_events()
# #         vis.update_renderer()
# #
# #         time.sleep(1.0)
# #
# # # vis.destroy_window()
# #
# # if __name__ == '__main__':
# #     t = threading.Thread(target=run)
# #     t.start()
# #     print('sadasd')
# #     t.join()
#

import cv2
import numpy as np

a = np.array([
    [0,1,2],
    [3,4,5],
    [6,7,8]
])

b = np.array([0,1,2])
print(a[b, b])

c = np.array([
    [0,0],
    [1,1],
    [2,2]
])
print(a[c])
