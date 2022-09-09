//
// Created by Tosin Oseni on 05/09/2022.
//

#ifndef STEREO_SLAM__G2O_TYPES_H
#define STEREO_SLAM__G2O_TYPES_H

#include "common_include.h"

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

namespace stereoSlam {


    // define vertex variables to be used G2o
    class VertexPose : public g2o::BaseVertex<6, SE3> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        //  intialise the override function where we set the estimate
        virtual void setToOriginImpl() override {
            _estimate = SE3();
        }


        /// left multiplication to SE3 to update pose
        virtual void oplusImpl(const double *update) override {
            Vec6 delta;
            delta << update[0], update[1], update[2], update[3], update[4], update[5];

            // estimate update is exp map(delta) * old pose
            _estimate = SE3::exp(delta) * _estimate;
        }

        // i belive these are functions needed by g2o interface functions of some sort
        virtual bool read(std::istream &in) override { return true; }

        virtual bool write(std::ostream &out) const override { return true; }

    };


/// Waypoint vertex
    class VertexXYZ : public g2o::BaseVertex<3, Vec3> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        // override state initialization
        virtual void setToOriginImpl() override {
            _estimate = Vec3::Zero();
            // 3d vector point
        }

        virtual void oplusImpl(const double *update) override {
            // follows state update none x = x + del x
            _estimate[0] += update[0];
            _estimate[1] += update[1];
            _estimate[2] += update[2];
        }


        // i belive these are functions needed by g2o interface functions of some sort
        virtual bool read(std::istream &in) override { return true; }

        virtual bool write(std::ostream &out) const override { return true; }

    };

/// Estimate only unary edges of the pose
    class EdgeProjectionPoseOnly : public g2o::BaseUnaryEdge<2, Vec2, VertexPose> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        // constructor
        EdgeProjectionPoseOnly(const Vec3 &pos, const Mat33 &K)
                : _pos3d(pos), _K(K) {}

        virtual void computeError() override {
            // force vertix
            const VertexPose *vertex_p = static_cast<VertexPose *>(_vertices[0]);
            SE3 T = vertex_p->estimate();

            // compute u,v = K *T * P
            // P is 3d point in world space using camera model
            Vec3 pos_pixel = _K * (T * _pos3d);

            // normlaise u,v /Z
            pos_pixel /= pos_pixel[2];
            // error between the u,v we measured vs the one estimated from 3d point
            _error = _measurement - pos_pixel.head<2>();
        }

        virtual void linearizeOplus() override {

            // function to define jacobian
            const VertexPose *v = static_cast<VertexPose *>(_vertices[0]);
            SE3 T = v->estimate();
            Vec3 pos_cam = T * _pos3d;
            double fx = _K(0, 0);
            double fy = _K(1, 1);

            // 3d point
            double X = pos_cam[0];
            double Y = pos_cam[1];
            double Z = pos_cam[2];

            // check derivation for bundle adjustmenrt
            double Zinv = 1.0 / (Z + 1e-18);
            double Zinv2 = Zinv * Zinv;

            _jacobianOplusXi << -fx * Zinv, 0, fx * X * Zinv2, fx * X * Y * Zinv2,
                    -fx - fx * X * X * Zinv2, fx * Y * Zinv, 0, -fy * Zinv,
                    fy * Y * Zinv2, fy + fy * Y * Y * Zinv2, -fy * X * Y * Zinv2,
                    -fy * X * Zinv;
        }


        virtual bool read(std::istream &in) override { return true; }

        virtual bool write(std::ostream &out) const override { return true; }

    private:
        Vec3 _pos3d;
        Mat33 _K;

    };


//    class EdgeProjection : public g2o::BaseBinaryEdge<2, Vec2, VertexPose, VertexXYZ> {
//    public:
//        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
//
//        EdgeProjection(const Mat33 &K, const SE3 &cam_ext) : _K(K) {
//            _cam_ext = cam_ext;
//        }
//        virtual bool read(std::istream &in) override { return true; }
//
//        virtual bool write(std::ostream &out) const override { return true; }
//
//    private:
//        // define memeber variables
//        SE3 _cam_ext;
//        Mat33 _K;
//
//    };

}
#endif //STEREO_SLAM__G2O_TYPES_H
