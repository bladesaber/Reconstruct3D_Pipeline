import numpy as np
import cv2
import apriltag

K = np.array([
    [606.96, 0, 326.85],
    [0, 606.10, 244.87],
    [0, 0, 1]
])
Kv = np.linalg.inv(K)


def compute_axis(corner, dst_points):
    homography, _ = cv2.findHomography(dst_points, corner, method=0)

    r1 = Kv.dot(homography[:, 0:1])
    r2 = Kv.dot(homography[:, 1:2])

    r1_length = np.sqrt(np.sum(np.power(r1, 2)))
    r2_length = np.sqrt(np.sum(np.power(r2, 2)))

    s = np.sqrt((r1_length * r2_length))

    t = Kv.dot(homography[:, 2])
    if t[-1] < 0:
        s = s * -1

    t = t / s
    r1 = r1 / s
    r2 = r2 / s

    r3 = (np.cross(r1.T, r2.T)).reshape((-1, 1))
    rot = np.concatenate((r1, r2, r3), axis=1)

    # U, S, V = np.linalg.svd(rot)
    # rot = U.dot(V)
    RT = np.concatenate((rot, t.reshape((-1, 1))), axis=1)

    return RT

uv_array = np.array([
    [188.33858, 79.35309],
    [494.52295, 95.80984],
    [552.5096, 326.21774],
    [124.27944, 304.73135]
])
dst_array = np.array([
    [-0.1485, 0.105],
    [0.1485, 0.105],
    [0.1485, -0.105],
    [-0.1485, -0.105]
])

RT = compute_axis(corner=uv_array, dst_points=dst_array)

opoints = np.array([
    ### center
    [0., 0., 0., 1.0],

    ### corner
    [-0.1485, 0.105, 0., 1.0],
    [0.1485, 0.105, 0., 1.0],
    [0.1485, -0.105, 0., 1.0],
    [-0.1485, -0.105, 0., 1.0],

    # ### x, y, z
    # [0.1, 0., 0., 1.0],
    # [0., 0.1, 0., 1.0],
    # [0., 0., 0.1, 1.0]

]).astype(np.float32)

K = np.concatenate([K, np.zeros((3, 1))], axis=1)
RT = np.concatenate((RT, np.array([[0., 0., 0., 1.]])), axis=0)
ipoints = (K.dot(RT.dot(opoints.T))).T
ipoints = ipoints[:, :2] / ipoints[:, -1:]

img = cv2.imread('/home/quan/Desktop/company/Reconstruct3D_Pipeline/location/test.jpg')
height, width, c = img.shape
resize_width = 640
resize_height = 480
img = cv2.resize(img, (resize_width, resize_height))

for p in ipoints:
    x, y = p
    x, y = int(x), int(y)
    cv2.circle(img, (x, y), 4, (255, 0, 0), 2)

cv2.imshow('d', img)
cv2.waitKey(0)