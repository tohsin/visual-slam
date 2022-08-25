#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include "common.h"
using namespace Sophus;
using namespace Eigen;
using namespace std;
int main() {
    std::cout << "Hello, World!" << std::endl;
    string file_path = "/Users/emma/dev/visual-slam/poseGraph_g2o/problem-16-22106-pre.txt";
    BALProblem bal_problem(file_path);
    bal_problem.Normalize();
    bal_problem.Perturb(0.1, 0.5, 0.5);
    return 0;
}
