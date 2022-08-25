#include <iostream>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel_impl.h>


#include "common.h"
#include "sophus/se3.hpp"

using namespace Sophus;
using namespace Eigen;
using namespace std;

struct PoseandIntristics{
    PoseandIntristics(){}

    /// set from given data address
    //explicit means you cant implicitly call constructor lkine class A = 22;
    // you have to call as a cast or class decleration
    explicit PoseandIntristics(double *data_addr){
        // rotatiion vector from phi
        rotation = SO3d::exp(  Vector3d(data_addr[0], data_addr[1], data_addr[2]) );
        // trasnaltion is just a vector
        translation = Vector3d(data_addr[3], data_addr[4], data_addr[5]);

        focal = data_addr[6];
        k1 = data_addr[7];
        k2 = data_addr[8];
    }


    // function to update data back by converting back to suitable valuess and updatting addresses
    void set_to(double *data_addr) {
        auto r = rotation.log();
        for (int i = 0; i < 3; ++i){
            data_addr[i] = r[i];
        }
        for (int i = 0; i < 3; ++i) {
            data_addr[i + 3] = translation[i];
        }
        data_addr[6] = focal;
        data_addr[7] = k1;
        data_addr[8] = k2;
    };
    SO3d rotation;
    Vector3d translation = Vector3d::Zero();
    double focal = 0;
    double k1 = 0, k2 = 0;
};


int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
