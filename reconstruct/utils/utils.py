from scipy.spatial import transform
import numpy as np

def quaternion_to_rotationMat_scipy(quaternion):
    r = transform.Rotation(quat=quaternion)
    return r.as_dcm()

def quaternion_to_eulerAngles_scipy(quaternion, degrees=False):
    r = transform.Rotation(quat=quaternion)
    return r.as_euler(seq='xyz', degrees=degrees)

def rotationMat_to_quaternion_scipy(R):
    r = transform.Rotation.from_dcm(dcm=R)
    return r.as_quat()

def rotationMat_to_eulerAngles_scipy(R, degrees=False):
    r = transform.Rotation.from_dcm(dcm=R)
    return r.as_euler(seq='xyz', degrees=degrees)

def eulerAngles_to_quaternion_scipy(theta, degress):
    r = transform.Rotation.from_euler(seq='xyz', angles=theta, degrees=degress)
    return r.as_quat()

def eulerAngles_to_rotationMat_scipy(theta, degress):
    r = transform.Rotation.from_euler(seq='xyz', angles=theta, degrees=degress)
    return r.as_dcm()

def rotationVec_to_rotationMat_scipy(vec):
    r = transform.Rotation.from_rotvec(vec)
    return r.as_dcm()

def rotationVec_to_quaternion_scipy(vec):
    r = transform.Rotation.from_rotvec(vec)
    return r.as_quat()

def xyz_to_ply(point_cloud, filename, rgb=None):
    if rgb is not None:
        colors = rgb.reshape(-1, 3)
        point_cloud = point_cloud.reshape(-1, 3)

        assert point_cloud.shape[0] == colors.shape[0]
        assert colors.shape[1] == 3 and point_cloud.shape[1] == 3

        vertices = np.hstack([point_cloud, colors])

        np.savetxt(filename, vertices, fmt='%f %f %f %d %d %d')  # 必须先写入，然后利用write()在头部插入ply header

        ply_header = '''ply
                    format ascii 1.0
                    element vertex %(vert_num)d
                    property float x
                    property float y
                    property float z
                    property uchar red
                    property uchar green
                    property uchar blue
                    end_header
                    \n
                    '''

        with open(filename, 'r+') as f:
            old = f.read()
            f.seek(0)
            f.write(ply_header % dict(vert_num=len(vertices)))
            f.write(old)

    else:
        point_cloud = point_cloud.reshape(-1, 3)

        assert point_cloud.shape[1] == 3

        np.savetxt(filename, point_cloud, fmt='%f %f %f')

        ply_header = '''ply
                        format ascii 1.0
                        element vertex %(vert_num)d
                        property float x
                        property float y
                        property float z
                        end_header
                        \n
                        '''

        with open(filename, 'r+') as f:
            old = f.read()
            f.seek(0)
            f.write(ply_header % dict(vert_num=len(point_cloud)))
            f.write(old)

