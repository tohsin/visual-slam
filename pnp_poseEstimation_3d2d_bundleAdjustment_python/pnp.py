from unicodedata import name

from scipy.linalg import lstsq

import sys
import os
import cv2 as cv
import numpy as np
import sophus as sp

def pixel_to_cam(point2d , k):
    return ((point2d[0] - k[0][2]) / k[0][0] , (point2d[1]- k[1][2])/k[1][1])

def find_feature_matches(img_1, img_2):
    orb_detector = cv.ORB_create()

    # find the keypoints with ORB
    kp1, des1 = orb_detector.detectAndCompute(img_1,None)
    kp2, des2 = orb_detector.detectAndCompute(img_2,None)

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    match = bf.match(des1,des2)
    print("number of descriptors",len(des2))
    # compute highest and lowest distance
    min_dist = sys.maxsize
    max_dist = - sys.maxsize
    for i in range(len(match)):
        dist = match[i].distance
        min_dist = min(min_dist, dist)
        max_dist = max(max_dist,  dist)
    print("max distance " ,max_dist )
    print("min distance :", min_dist)
    print("number of matches",len(match))

    matches = []
    for i in range(len(match)):
        if match[i].distance <= max(2 * min_dist, 30.0) :
            matches.append(match[i])
    return matches, kp1, kp2

def bundleAdjustmentGaussNewton(pts_3d, pts_2d, K):
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]
    cost = 0
    last_cost = 0
    pose = sp.SE3()


    n_iter = 10
    for iter in range(n_iter):
        H = np.zeros((6,6))
        b = np.zeros((6,1))
        cost  = 0
        
        for i in range(len(pts_3d)):
            projected_camera = pose *  np.array(pts_3d[i])
            X = projected_camera[0]
            Y = projected_camera[1]
            Z = projected_camera[2]
            inverse_z = 1.0 / Z
            inverse_z_squared = inverse_z * inverse_z

            camera_project_point = np.array([fx * X / Z + cx, fy * Y / Z + cy])
            error = pts_2d[i] - camera_project_point
            
            chi_error = np.dot(error,error)

            
            # mse= np.linalg.norm(veceerortor)
            cost += chi_error
            jacobian = [
                [-fx * inverse_z,
                0 , 
                fx * X * inverse_z_squared, 
                fx * X * Y* inverse_z_squared , 
                -fx - fx * X * X * inverse_z_squared, 
                fx *Y*inverse_z],
                [0, 
                -fy * inverse_z, 
                fy * Y * inverse_z_squared,
                fy + fy * Y *Y * inverse_z_squared,
                -fy*X *Y*inverse_z_squared,
                -fy* X*inverse_z]
            ]
            jacobian  = np.array(jacobian)
            # print(jacobian.shape)
            # print(jacobian.T.shape)

            H += np.dot(jacobian.T , jacobian)
            b_value = np.dot(jacobian.T ,error)
       
            b_value = b_value.reshape(6,1)
            b += b_value

        dx = np.linalg.lstsq(H, -b, rcond=None)[0]

        # if  dx :
        #     print("result is none ")
        #     break
        if iter>0 and cost>=last_cost:
            print("cost ", cost, "Last cost", last_cost)

        exp_ = sp.SE3.exp(dx)
        print("exp", exp_)
        r_matrix = matrix_exp_rotation(dx[3:])
        translation = matrix_exp_translation(dx[3:], dx[:3])
        T_matrix = matrix_exp(r_matrix, translation)
        # print("Transformation matrix ",T_matrix)
        # break
       
        # pose =  sp.SE3.exp(dx) * pose
        pose = sp.SE3(T_matrix) * pose
        last_cost = cost
        
        print("iteration ", iter," cost = ", cost)
        dx_mean = (dx).mean(axis=0)
        value_mean = dx_mean[0]
        dx_norm = np.sqrt(abs(value_mean))
        if (dx_norm < 1e-6):
            break
    print(pose)

def matrix_exp_rotation(omega):
    omega = np.array(omega)
    R = cv.Rodrigues(omega)[0]
    # print("Demo R")
    # print(R)
    
    theta =  np.linalg.norm(omega)
    theta_sq = theta * theta
    factor_1 = np.sin(theta)/ theta
    factor_2 = (1- np.cos(theta)) / theta_sq
    omega_1 = omega[0]
    omega_2 = omega[1]
    omega_3 = omega[2]
    skew_matrix = [
        [0, -omega_3, omega_2],
        [omega_3, 0 , -omega_1],
        [-omega_2, omega_1, 0]
   ]
    skew_matrix = np.array(skew_matrix)
    skew_theta_squared = np.dot(skew_matrix,skew_matrix)
    I = np.identity(3)
    rotation_matrix = I + factor_1 * skew_matrix + factor_2 *skew_theta_squared
    rotation_matrix.tolist()
    r_1 = rotation_matrix[0].tolist()
    r_2 = rotation_matrix[1].tolist()
    r_3 = rotation_matrix[2].tolist()

    r_mat = [ 
        [r_1[0].tolist()[0], r_1[1].tolist()[0] , r_1[2].tolist()[0]], 
        [r_2[0].tolist()[0], r_2[1].tolist()[0] , r_2[2].tolist()[0]], 
        [r_3[0].tolist()[0], r_3[1].tolist()[0] , r_3[2].tolist()[0]], 
        ]
    # print("Here")
  
    
    
    return np.array(r_mat)
  
def matrix_exp_translation(omega, rho):
    theta =  np.linalg.norm(omega)
    theta_sq = theta * theta
    theta_cube = theta * theta *theta
    omega_1 = omega[0]
    omega_2 = omega[1]
    omega_3 = omega[2]

    skew_matrix = [
        [0, -omega_3, omega_2],
        [omega_3, 0 , -omega_1],
        [-omega_2, omega_1, 0]
   ]
    skew_matrix = np.array(skew_matrix)
    skew_theta_squared = np.dot(skew_matrix,skew_matrix)
    factor_1 =  (1 - np.cos(theta))/ theta_sq
    factor_2 = (theta - np.sin(theta)) / theta_cube


    I = np.identity(3)
    jacobian =  I + (factor_1 * skew_matrix) + (factor_2 * skew_theta_squared)
    jacobian.tolist()
    r_1 = jacobian[0].tolist()
    r_2 = jacobian[1].tolist()
    r_3 = jacobian[2].tolist()

    j_mat = [ 
        [r_1[0].tolist()[0], r_1[1].tolist()[0] , r_1[2].tolist()[0]], 
        [r_2[0].tolist()[0], r_2[1].tolist()[0] , r_2[2].tolist()[0]], 
        [r_3[0].tolist()[0], r_3[1].tolist()[0] , r_3[2].tolist()[0]], 
        ]
    j_mat = np.array(j_mat)
    jp = np.dot(j_mat , rho)
    print(jp)
    return jp

def matrix_exp(rotation, translation):
    return np.array([
        [rotation[0][0], rotation[0][1], rotation[0][2],translation[0] ],
        [rotation[1][0], rotation[1][1], rotation[1][2],translation[1]],
        [rotation[2][0],rotation[2][1],rotation[2][2], translation[2]],
        [0,0,0,1]
    ])



if __name__ == '__main__':
    base_dir = os.getcwd()
    image1 = os.path.join(base_dir, "pnp/1.png")
    image2 = os.path.join(base_dir,"pnp/2.png")
    image3 = os.path.join(base_dir,"pnp/1_depth.png")


    image1 = cv.imread(image1, cv.IMREAD_COLOR )
    image2 = cv.imread(image2, cv.IMREAD_COLOR )

    keypoints_1 = []
    keypoints_2 = []
    matches = []
    matches , keypoints_1, keypoints_2= find_feature_matches(image1, image2)


    depth_image = cv.imread(image3, cv.IMREAD_UNCHANGED)
    print("depth image shape", depth_image.shape)
    K = [[ 520.9, 0, 325.1], \
        [0, 521.0, 249.7],\
        [ 0, 0, 1]]
    pts_3d = []
    pts_2d = []

    for match in matches:
        idx = int(keypoints_1[match.queryIdx].pt[0])
        idy = int(keypoints_1[match.queryIdx].pt[1])
        d = depth_image[idy][idx]
        if d ==0:
            continue
        dd = d/5000.0
        p1 = pixel_to_cam(keypoints_1[match.queryIdx].pt, K)

        pos_3d = [p1[0] * dd, p1[1] * dd,  dd]
        pts_3d.append(pos_3d)

        pts_2d.append(keypoints_2[match.trainIdx].pt)
    print("number of 3d-2d pairs", len(pts_3d), len(pts_2d))

    distortion_coeffs = np.zeros((4,1))
    pts_3d_np = np.array(pts_3d)
    pts_2d_np = np.array(pts_2d)
    K_np = np.array(K)


    success, vector_rotation, vector_translation = cv.solvePnP(pts_3d_np, pts_2d_np, K_np,distortion_coeffs, flags=0,useExtrinsicGuess=False)
    print("rotation matrix", vector_rotation)
    print("translation vector",vector_translation )
    print(vector_rotation.shape)
   
    R =cv.Rodrigues(vector_rotation)
  
    print("R from rodrigues", R)

    # coputing bundle adjustent
    bundleAdjustmentGaussNewton(pts_3d_np, pts_2d_np, K)