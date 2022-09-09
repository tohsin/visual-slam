//
// Created by Tosin Oseni on 08/09/2022.
//
#include "stereo_slam/backend.h"
#include "stereo_slam/g2o_types.h"

namespace stereoSlam{

    Backend::Backend() {
        backend_running_.store(true);
        backend_thread_ = std::thread(std::bind(&Backend::BackendLoop, this));
    }


    void Backend::Stop() {
        // set state of back end running
        backend_running_.store(false);
        // update state of map_update to allow next step of
        map_update_.notify_one();
        //If any threads are waiting on *this, calling notify_one unblocks one of the waiting threads.
        backend_thread_.join();
    }

    void Backend::UpdateMap() {
        std::unique_lock<std::mutex> lock(data_mutex_);
        map_update_.notify_one();
    }
    void stereoSlam::Backend::Optimize(Map::KeyframesType &keyframes, Map::LandmarksType &landmarks) {
        typedef g2o::BlockSolver_6_3 BlockSolverType; // bA solver
        typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;

        auto solver = new g2o::OptimizationAlgorithmLevenberg(
                g2o::make_unique<BlockSolverType>(
                        g2o::make_unique<LinearSolverType>()));

//       typedef g2o::BlockSolver_6_3 BlockSolverType;
//        typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;
//        auto solver = new g2o::OptimizationAlgorithmLevenberg(
//            g2o::make_unique<BlockSolverType>(
//            g2o::make_unique<LinearSolverType>()));
//        g2o::SparseOptimizer optimizer;
//        optimizer.setAlgorithm(solver);

//        // to store verices with their index and vertex pose nodes
//        std::map<unsigned long, VertexPose *> vertices;
//        unsigned long max_keyfame_id = 0;
//        for(auto &keyframe : keyframes){
//            auto kf = keyframe.second; // map id and frame
//            VertexPose* vertex_pose = new VertexPose();
//            vertex_pose->setId(kf->key_frame_id_);
//            vertex_pose->setEstimate(kf->Pose());
//            optimizer.addVertex(vertex_pose);
//            if (kf->key_frame_id_ > max_keyfame_id) {
//                max_keyfame_id = kf->key_frame_id_;
//            }
//            vertices.insert({kf->key_frame_id_, vertex_pose});
//        }
//
//        // Waypoint vertex, indexed by waypoint id
//        std::map<unsigned long, VertexXYZ *> vertices_landmarks;
//
//
//        // left and right external parameters
//        Mat33 K = cam_left_->K();
//        SE3 left_external_parameters = cam_left_->pose();
//        SE3 right_external_parameters = cam_right_->pose();
//
//        int index = 1;
//        double chi2_th = 5.991;  // robust kernel threshold
//        std::map<EdgeProjection *, Feature::Ptr> edges_and_features;

    }
    void Backend::BackendLoop() {
        // run infinite loop to run program
        while(backend_running_.load()){
            std::unique_lock<std::mutex> lock(data_mutex_);
            map_update_.wait(lock);

            // acesss key frames and landmarks
            Map::KeyframesType active_keyframes = map_->GetActiveKeyFrames();
            Map::LandmarksType active_landmarks = map_->GetActiveMapPoints();
            Optimize(active_keyframes, active_landmarks);
        }


    }


}
