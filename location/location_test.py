import cv2
import argparse
from queue import Queue
import time

from camera.camera_realsense import Camera_RealSense
from location.posefinder import Calibration_PoseFinder
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ### left topic listener param
    parser.add_argument('--coodr_path', type=str,
                        default='/home/quan/Desktop/company/Reconstruct3D_Pipeline/location/coodr.txt')
    args, _ = parser.parse_known_args()
    return args

### ----------------------------------------------------------------
def opencv_event(event, x, y, flags, param):
    global mouse_x, mouse_y
    # print('[DEBUG]: Mouse X: ', x, ' Mouse Y:', y)
    mouse_x = x
    mouse_y = y

### -------- global param
mouse_x, mouse_y = 0.0, 0.0

### -------------------------
def calibration_RT(camera):

    ### ------ init param
    global mouse_x, mouse_y
    coodr_queue = Queue(maxsize=4)
    detection_corners = np.array([])
    select_index = None
    RT = None

    args = parse_args()

    pose_finder = Calibration_PoseFinder(coodr_path=args.coodr_path)

    cv2.namedWindow('debug')
    cv2.setMouseCallback('debug', opencv_event)

    while True:
        status, rgb_img, _ = camera.get_img()

        if status:
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

            # gray = pose_finder.color_thresold(img=rgb_img, is_rgb=True, r_thresold=180, g_thresold=180, b_thresold=180)
            gray = pose_finder.to_gray(rgb_img, is_rgb=True)
            gray = pose_finder.adaptiveThreshold(img=gray, block_size=7)
            detection_corners = pose_finder.detect_harris(gray=gray, score=130, blockSize=3)

            rgb_img = pose_finder.draw_points(img=rgb_img, points_2d=detection_corners)
            if detection_corners.shape[0] > 0:
                current_pos = np.array([[mouse_x, mouse_y]])

                distance_square = np.sum(np.power(detection_corners - current_pos, 2), axis=1)
                select_index = np.argmin(distance_square)

                point_x = int(detection_corners[select_index][0])
                point_y = int(detection_corners[select_index][1])
                cv2.circle(rgb_img, (point_x, point_y), 4, (255, 0, 0), 2)

            if RT is not None:
                pose_finder.draw_rect(rgb=rgb_img, RT=RT)

            cv2.imshow('debug', rgb_img)

        key = cv2.waitKey(5)
        if key == ord('q'):
            break

        elif key == ord('s'):
            coodr_queue.put(detection_corners[select_index])
            print('[DEBUG]: Adding Corner: ', detection_corners[select_index])

        elif key == ord('c'):

            uv_array = []
            while not coodr_queue.empty():
                uv_array.append(coodr_queue.get())
            uv_array = np.array(uv_array)

            print('[DEBUG]UV Matrix:\n', uv_array)
            print('[DEBUG]Coodr Matrix:\n', pose_finder.coodr_prefix)
            print('\n')

            assert uv_array.shape[0] == pose_finder.coodr_prefix.shape[0]
            RT = pose_finder.compute_axis(corner=uv_array, dst_points=pose_finder.coodr_prefix)
            # cv2.solvePnP(
            #     objectPoints=pose_finder.coodr_prefix, imagePoints=uv_array,
            #     cameraMatrix=pose_finder.K, distCoeffs=pose_finder.dcoeffs, flags=cv2.SOLVEPNP_ITERATIVE
            # )

    cv2.destroyAllWindows()

    return RT

def object_pose_detection(camera, RT):
    while True:
        status, rgb_img, _ = camera.get_img()

        if status:
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            cv2.imshow('debug', rgb_img)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

if __name__ == '__main__':
    camera = Camera_RealSense()

    RT = calibration_RT(camera=camera)
    print('[DEBUG] RT:\n', RT)

    object_pose_detection(camera=camera, RT=RT)


