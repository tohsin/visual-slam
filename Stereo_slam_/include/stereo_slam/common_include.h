//
// Created by Tosin Oseni on 26/08/2022.
//

#ifndef STEREO_SLAM__COMMON_INCLUDE_H
#define STEREO_SLAM__COMMON_INCLUDE_H


#include <iostream>
#include <vector>
#include <thread>
#include <typeinfo>
#include <list>
#include <unordered_map>
#include <memory> // for smart pointers


// define the commonly included file to avoid a long include list
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>


using cv::Mat;

// for Sophus
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

typedef Sophus::SE3d SE3;
typedef Sophus::SO3d SO3;

// vector defintions
typedef Eigen::Matrix<double, 6, 1> Vec6;
typedef Eigen::Matrix<double, 3, 1> Vec3;
typedef Eigen::Matrix<double, 2, 1> Vec2;

// matrix definition
typedef Eigen::Matrix<double, 3, 3> Mat33; // 3 by 3 matrix
#endif //STEREO_SLAM__COMMON_INCLUDE_H
