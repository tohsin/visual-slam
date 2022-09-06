//
// Created by Tosin Oseni on 26/08/2022.
//
# pragma  once
#ifndef STEREO_SLAM__CAMERA_H
#define STEREO_SLAM__CAMERA_H

#include "stereo_slam/common_include.h"

namespace stereoSlam {
class Camera{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Camera> Ptr;

    double fx_ = 0, fy_ = 0, cx_ = 0, cy_ = 0, baseline_ =0; // camera intrinsics


    SE3 pose_;       // extrinsic, from stereo camera to single camera
    SE3 pose_inv_;   // inverse of extrinsics
    Camera();

    Camera(double fx, double fy, double cx, double cy, double baseline,
           const SE3 &pose)
            : fx_(fx), fy_(fy), cx_(cx), cy_(cy), baseline_(baseline), pose_(pose) {

        // compute pose inverse look up inverse of T
        /*
         T_inv = [ R.T    - R.T * t]
                 [ 0          1    ]
         */
//        Mat Rotation_mat = pose_.rowRange(0,3).colRange(0,3);
//        Mat translation_mat = pose_.rowRange(0,3).col(3);
//        Mat Rotation_mat_trasnpose = Rotation_mat.t();
//        Mat new_trasnpose = -Rotation_mat_trasnpose * translation_mat;
//        pose_inv_ = cv::Mat::eye(4,4,pose_.type()); // declare data structure to store T inverse
//
//        Rotation_mat_trasnpose.copyTo(pose_inv_.rowRange(0,3).colRange(0,3));
//        new_trasnpose.copyTo(pose_inv_.rowRange(0,3).col(3));
        pose_inv_ = pose.inverse();
    }
    SE3 pose() const { return pose_; }

    // return instristic matrix
    Mat33 K() const{
        Mat33 k;
        k << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1;
        return k;
    }


    // coordinate transform: world, camera, pixel
    Vec3 world2camera(const Vec3 &p_w, const SE3 &T_c_w);

    Vec3 camera2world(const Vec3 &p_c, const SE3 &T_c_w);

    Vec2 camera2pixel(const Vec3 &p_c);

    Vec3 pixel2camera(const Vec2 &p_p, double depth = 1);

    Vec3 pixel2world(const Vec2 &p_p, const SE3 &T_c_w, double depth = 1);

    Vec2 world2pixel(const Vec3 &p_w, const SE3 &T_c_w);


};
}

#endif //STEREO_SLAM__CAMERA_H
