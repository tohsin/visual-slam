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
# window_size = 15
# min_disp = 16
# num_disp = 96 - min_disp
stereo = cv.StereoSGBM_create(minDisparity=0,
                                   numDisparities=96,
                                   blockSize=9,
                                   P1=8 * 9 * 9,
                                   P2=32 * 9 *9,
                                   disp12MaxDiff=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=32
                                   )
h, w = left.shape[:2]
                          
f = 0.8 * w  # guess for focal length
Q = np.float32([[1, 0, 0, -0.5 * w],
                    [0, -1, 0, 0.5 * h],  # turn points 180 deg around x-axis,
                    [0, 0, 0, -f],  # so that y-axis looks up
                    [0, 0, 1, 0]])

disparity = stereo.compute(left, right)

disparity =  np.float32(np.divide(disparity, 16.0))

point3d =  cv.reprojectImageTo3D(disparity, Q, handleMissingValues=False)
colors = cv.cvtColor(left, cv.COLOR_BGR2RGB)

# get rid of minimum values in first place liel 0 or -16
mask_map = disparity > disparity.min()

out_point = point3d[mask_map]
out_colors =  colors[mask_map]

def generate_point_cloud(vertices, colors, file_name):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1,3), colors])
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
		'''
    with open(file_name, 'w') as f:
        f.write(ply_header %dict(vert_num=len(vertices)))
        np.savetxt(f,vertices,'%f %f %f %d %d %d')

output_file = os.path.join(base_dir,'pointCloud.ply')

# Generate point cloud file
# generate_point_cloud(out_point, out_colors, output_file)



def showPointCloud(point_cloud_dir):
   
    # stereo_file  = os.path.join(base_dir, "stereo_vision_python/data/stereo.ply")
    pcd = o3d.io.read_point_cloud(point_cloud_dir)
    o3d.visualization.draw_geometries([pcd])
showPointCloud("/Users/emma/dev/visual-slam/stereo_vision_python/data/pointCloud.ply")


