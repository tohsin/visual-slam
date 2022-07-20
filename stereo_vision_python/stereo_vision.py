from email.mime import base
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import OpenGL.GL as gl
# import pangolin
import open3d as o3d
import os


def showPointCloud(point_cloud):
    pcd = o3d.geometry.PointCloud()
    np_array = np.array(point_cloud)
    pcd.points = o3d.utility.Vector3dVector(np_array)
    stereo_file  = os.path.join(base_dir, "stereo_vision_python/data/stereo.ply")
    o3d.io.write_point_cloud(stereo_file, pcd)

    stereo_file  = os.path.join(base_dir, "stereo_vision_python/data/stereo.ply")
    pcd = o3d.io.read_point_cloud(stereo_file)
    o3d.visualization.draw_geometries([pcd])

fx = 718.856
fy = 718.856
cx = 607.1928
cy = 185.2157

base_line = 0.573
base_dir = os.getcwd()

left_image = os.path.join(base_dir, "stereo_vision_python/data/left.png")
right_image = os.path.join(base_dir,"stereo_vision_python/data/right.png")


left = cv.imread(left_image,0)
right = cv.imread(right_image, 0)


sgbm = cv.StereoBM_create(numDisparities=96, blockSize= 9)
sgbm.setMinDisparity(0)
sgbm.setUniquenessRatio(10)
sgbm.setPreFilterCap(63)
sgbm.setSpeckleRange(32)
sgbm.setSpeckleWindowSize(100)
sgbm.setDisp12MaxDiff(1)


disparity_sgbm : cv.Mat
disparity : cv.Mat

disparity = sgbm.compute(left, right)

point_cloud = []
disparity.astype(np.float32)
# print(disparity.tolist())

# print(left.shape[0])


for v in range(left.shape[0]):
    for u in range(left.shape[1]):
       
        if disparity[v][u] <= 10.0 or disparity[v][u]>= 96.0:
            continue
        # print(disparity[v][u])
        # print(disparity[v][u])
        
        point = [0, 0, 0 ] # x y z color
        x = ( u - cx )/ fx
        y = (v - cy)/  fy
        depth = (fx * base_line) / disparity[v][u]
        point[0] = x * depth
        point[1] = y * depth
        point[2] = depth
        
        point_cloud.append(point)

showPointCloud(point_cloud)




o3d.visualization.draw_geometries([pcd])
plt.imshow(disparity/96.0,'gray')
plt.show()




