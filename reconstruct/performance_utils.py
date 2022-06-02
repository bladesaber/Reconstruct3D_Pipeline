import open3d
import numpy as np
import time
import threading
import cv2

class NonBlock_Visualizer(object):
    def __init__(self):
        self.vis = open3d.visualization.Visualizer()
        self.vis.create_window()

        self.vis_pcd = open3d.geometry.PointCloud()
        colors = np.tile(np.array([[255, 0, 0]], dtype=np.uint8), (1000, 1))
        points = np.random.random((1000, 3))
        self.vis_pcd.points = open3d.utility.Vector3dVector(points)
        self.vis_pcd.colors = open3d.utility.Vector3dVector(colors)

        self.vis.add_geometry(self.vis_pcd)

        self.threat = threading.Thread(target=self.visualizer_update)

    def visualizer_update(self):
        while True:
            self.vis.update_geometry(self.vis_pcd)
            self.vis.poll_events()
            self.vis.update_renderer()

            time.sleep(0.1)

    def start(self):
        self.threat.start()

    def pcd_update(self):
        raise ValueError