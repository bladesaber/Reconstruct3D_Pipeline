import os

import pyrealsense2 as rs
import numpy as np
import cv2
import time
import json

class Camera_RealSense(object):
    def __init__(
            self,
            input_bag_file=None,

            # record_bag_file='/home/quan/Desktop/tempary/2022_05_30.bag',
            record_bag_file=None,

            enable_color=True, enable_depth=True,
            enable_ir_left=False, enable_ir_right=False,
            width=640, height=480,
            color_format=rs.format.bgr8,
            depth_format=rs.format.z16,
            ir_format=rs.format.y8,
            fps=15,

            save_dir=''
    ):
        self.enable_color = enable_color
        self.enable_depth = enable_depth
        self.enable_ir_left = enable_ir_left
        self.enable_ir_right = enable_ir_right
        self.width = width
        self.height = height
        self.color_format = color_format
        self.depth_format = depth_format
        self.ir_format = ir_format

        self.fps = fps
        self.cost_time_avg = 0.0
        self.time_count = 0.0

        if input_bag_file is not None:
            self.read_bag = True
        else:
            self.read_bag = False

        self.pipeline = rs.pipeline()

        # Configure streams
        self.config = rs.config()
        if self.read_bag:
            self.config.enable_device_from_file(input_bag_file)
            print('[Debug]: Staring Reading the Bag ...')
        if record_bag_file is not None:
            self.config.enable_record_to_file(record_bag_file)
            print('[Debug]: Staring Record the Bag ...')

        if self.enable_depth:
            self.config.enable_stream(rs.stream.depth, self.width, self.height, self.depth_format, self.fps)
        if self.enable_color:
            self.config.enable_stream(rs.stream.color, self.width, self.height, self.color_format, self.fps)
        if self.enable_ir_left:
            self.config.enable_stream(rs.stream.infrared, 1, self.width, self.height, self.ir_format, self.fps)
        if self.enable_ir_right:
            self.config.enable_stream(rs.stream.infrared, 2, self.width, self.height, self.ir_format, self.fps)

        self.pipeline_profile = self.pipeline.start(self.config)

        self.filter_init()

        # self.depth_scale = self.pipeline_profile.get_device().first_depth_sensor().get_depth_scale()
        self.depth_scale = 1000.0
        print("[Debug]: Depth Scale is: ", self.depth_scale)

        self.intrinsics = {}

        self.color_save_dir = os.path.join(save_dir, 'color')
        self.depth_save_dir = os.path.join(save_dir, 'depth')
        self.intrinsics_path = os.path.join(save_dir, 'instrincs.json')
        if not os.path.exists(self.color_save_dir):
            os.mkdir(self.color_save_dir)
        if not os.path.exists(self.depth_save_dir):
            os.mkdir(self.depth_save_dir)

    def filter_init(self, decimation_magnitude=1.0,
                    spatial_magnitude=2.0, spatial_smooth_alpha=0.5,
                    spatial_smooth_delta=20,
                    temporal_smooth_alpha=0.4, temporal_smooth_delta=20):

        self.colorizer = rs.colorizer()

        ### todo ???
        ### 对准
        # self.align_ir = rs.align(rs.stream.infrared)
        self.align_color = rs.align(rs.stream.color)

        self.decimation_filter = rs.decimation_filter()
        self.spatial_filter = rs.spatial_filter()
        self.temporal_filter = rs.temporal_filter()

        filter_magnitude = rs.option.filter_magnitude
        filter_smooth_alpha = rs.option.filter_smooth_alpha
        filter_smooth_delta = rs.option.filter_smooth_delta

        # Apply the control parameters for the filter
        self.decimation_filter.set_option(filter_magnitude, decimation_magnitude)
        self.spatial_filter.set_option(filter_magnitude, spatial_magnitude)
        self.spatial_filter.set_option(filter_smooth_alpha, spatial_smooth_alpha)
        self.spatial_filter.set_option(filter_smooth_delta, spatial_smooth_delta)
        self.temporal_filter.set_option(filter_smooth_alpha, temporal_smooth_alpha)
        self.temporal_filter.set_option(filter_smooth_delta, temporal_smooth_delta)

    def get_instrics(self):
        if len(self.intrinsics)==0:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align_color.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            depth_instrics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
            color_instrics = rs.video_stream_profile(color_frame.profile).get_intrinsics()

            self.intrinsics['color'] = {
                'cx': color_instrics.ppx,
                'cy': color_instrics.ppy,
                'fx': color_instrics.fx,
                'fy': color_instrics.fy,
                'width': color_instrics.width,
                'height': color_instrics.height,
            }
            self.intrinsics['depth'] = {
                'cx': depth_instrics.ppx,
                'cy': depth_instrics.ppy,
                'fx': depth_instrics.fx,
                'fy': depth_instrics.fy,
                'width': depth_instrics.width,
                'height': depth_instrics.height,
            }

        return self.intrinsics

    def start_debug(self):
        while True:

            start_time = time.time()

            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align_color.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape
            print('[Debug]: color_image:', color_colormap_dim,' depth_image:',depth_colormap_dim)

            images = np.hstack((color_image, depth_colormap))
            cv2.imshow('RealSense', images)
            key = cv2.waitKey(1)
            if key==ord('q'):
                break

            time_cost = time.time() - start_time
            self.cost_time_avg = (self.cost_time_avg * self.time_count + time_cost)/(self.time_count + 1.0)
            self.time_count += 1
            print('[Debug]: fps estimate ', 1.0/self.cost_time_avg)

        self.stop()

    def stop(self):
        self.pipeline.stop()

    def get_img(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align_color.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            return False, None, None

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        return True, color_image, depth_image

    def save_to_color_depth(self, color_img, depth_img, id):
        color_path = os.path.join(self.color_save_dir, '%d.jpg'%id)
        depth_path = os.path.join(self.depth_save_dir, '%d.png'%id)

        # depth_img[depth_img>255] = 0
        # depth_img = depth_img.astype(np.uint8)

        cv2.imwrite(color_path, color_img)
        cv2.imwrite(depth_path, depth_img)

    def save_instrincs(self, instrics):
        with open(self.intrinsics_path, 'w') as f:
            json.dump(instrics, f)

    def load_instrincs(self):
        with open(self.intrinsics_path, 'r') as f:
            instrics = json.load(f)
        return instrics

if __name__ == '__main__':
    camera = Camera_RealSense(
        input_bag_file='/home/quan/Desktop/tempary/dataset/2022_05_30.bag',
        save_dir='/home/quan/Desktop/tempary/dataset'
    )
    camera.start_debug()
