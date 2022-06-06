import os
import cv2
import json

class Camera_Fake_1(object):
    def __init__(self, save_dir, start_id=0):
        self.save_dir = save_dir
        self.color_save_dir = os.path.join(save_dir, 'color')
        self.depth_save_dir = os.path.join(save_dir, 'depth')

        self.img_max_num = max([int(name.replace('.jpg', '')) for name in os.listdir(self.color_save_dir)])
        self.run_id = start_id

    def get_img(self):
        color_path = '%d.jpg'%self.run_id
        depth_path = '%d.png'%self.run_id

        color_path = os.path.join(self.color_save_dir, color_path)
        depth_path = os.path.join(self.depth_save_dir, depth_path)
        print(color_path)

        color_image = cv2.imread(color_path)
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        if self.run_id<self.img_max_num:
            self.run_id += 1
            return True, color_image, depth_image
        else:
            return False, None, None

    def load_instrincs(self, intrinsics_path):
        with open(intrinsics_path, 'r') as f:
            instrics = json.load(f)
        return instrics

import os
import cv2

class Camera_Fake_2(object):
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.color_save_dir = os.path.join(save_dir, 'rgb')
        self.depth_save_dir = os.path.join(save_dir, 'depth')

        self.id_list = []

        self.color_files_dict = {}
        for file in os.listdir(self.color_save_dir):
            id = int(file.split('-')[0])
            self.color_files_dict[id] = file
            self.id_list.append(id)

        self.depth_files_dict = {}
        for file in os.listdir(self.depth_save_dir):
            id = int(file.split('-')[0])
            self.depth_files_dict[id] = file

        self.id_list = sorted(self.id_list)
        self.run_id = 0
        self.img_max_num = len(self.id_list)

    def get_img(self):
        get_id = self.id_list[self.run_id]
        color_path = self.color_files_dict[get_id]
        depth_path = self.depth_files_dict[get_id]

        color_path = os.path.join(self.color_save_dir, color_path)
        depth_path = os.path.join(self.depth_save_dir, depth_path)
        print(color_path)

        color_image = cv2.imread(color_path)
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        if self.run_id<self.img_max_num:
            self.run_id += 1
            return True, color_image, depth_image
        else:
            return False, None, None

    def get_img_from_range(self, start_id, end_id):
        img_pack = {}
        idx_list = []

        for idx in range(start_id, end_id+1, 1):
            color_path = self.color_files_dict[idx]
            depth_path = self.depth_files_dict[idx]

            color_path = os.path.join(self.color_save_dir, color_path)
            depth_path = os.path.join(self.depth_save_dir, depth_path)
            print(color_path)

            color_image = cv2.imread(color_path)
            depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

            img_pack[idx] = {
                'color': color_image, 'depth': depth_image
            }

            idx_list.append(idx)

        return img_pack, img_pack

    def load_instrincs(self, intrinsics_path):
        with open(intrinsics_path, 'r') as f:
            instrics = json.load(f)
        return instrics

if __name__ == '__main__':
    # camera = Camera_Fake_2(save_dir='/home/quan/Desktop/template/redwood-3dscan/data/rgbd/00003')
    #
    # status, color_img, depth_img = camera.get_img()
    # # print(color_img.shape)
    # # print(depth_img.shape)
    # print(depth_img.max())
    # # cv2.imshow('c', color_img)
    # # cv2.imshow('d', depth_img)
    # # cv2.waitKey(0)

    pass