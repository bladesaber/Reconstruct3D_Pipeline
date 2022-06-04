import open3d
import numpy as np
import time
import threading
import cv2

class NonBlock_Visualizer(object):
    def __init__(self, without_nvidia=False):
        self.vis = open3d.visualization.Visualizer()
        self.vis.create_window()

        self.vis_pcd = open3d.geometry.PointCloud()
        colors = np.tile(np.array([[255, 0, 0]], dtype=np.uint8), (1000, 1))
        points = np.random.random((1000, 3))
        self.vis_pcd.points = open3d.utility.Vector3dVector(points)
        self.vis_pcd.colors = open3d.utility.Vector3dVector(colors)

        self.vis.add_geometry(self.vis_pcd)

        if without_nvidia:
            self.threat = threading.Thread(target=self.visualizer_update)

        self.update_once()

    ### only work when there is no Nvidia
    def visualizer_update(self):
        while True:
            self.update_once()
            time.sleep(0.1)

    def update_once(self):
        self.vis.update_geometry(self.vis_pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

    def start(self):
        self.threat.start()

    def pcd_update(self):
        raise ValueError

    def update_until_time(self, second):
        start_time = time.time()
        while True:
            self.update_once()
            time.sleep(0.05)

            if time.time() - start_time>second:
                break

if __name__ == '__main__':
    vis = NonBlock_Visualizer(without_nvidia=False)

    for _ in range(100):
        pcd_current = open3d.geometry.PointCloud()
        colors = (np.random.randint(0, 255, (2000, 3))).astype(np.uint8)
        points = np.random.random((2000, 3))
        pcd_current.points = open3d.utility.Vector3dVector(points)
        pcd_current.colors = open3d.utility.Vector3dVector(colors)

        vis.vis_pcd.points = pcd_current.points
        vis.vis_pcd.colors = pcd_current.colors
        vis.update_once()
        time.sleep(0.1)

    pass