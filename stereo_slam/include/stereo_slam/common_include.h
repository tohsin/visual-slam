# pragma once

/*
File to be used in all incolusion 
*/

#include <iostream>
#include <vector>
#include <thread>
#include <typeinfo>
#include <list>
#include <memory> // for smart pointers


// define the commonly included file to avoid a long include list
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>

typedef Eigen::Matrix<double, 3, 1> Vec3;