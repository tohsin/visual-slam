//
// Created by Tosin Oseni on 26/08/2022.
//
#include <opencv2/opencv.hpp>
#include "stereo_slam/frontend.h"
#include "stereo_slam/g2o_types.h"
#include "stereo_slam/algorithm.h"
#include "stereo_slam/viewer.h"
namespace stereoSlam{

    bool Frontend::AddFrame(stereoSlam::Frame::Ptr frame){
        // impliment adding frame
        current_frame_ = frame;
        switch (status_)
        {
            case FrontendStatus::INITING:
                /* code */
                SteroInit();
                break;
            case FrontendStatus::TRACKING_GOOD:
                /* code */
                break;
            case FrontendStatus::TRACKING_BAD:
                /* code */
                Track();
                break;
            case FrontendStatus::LOST:
                Reset();
                break;
            default:
                break;

        }
        last_frame_ = current_frame_;
        return true;
    }

    bool Frontend::SteroInit(){
        int num_features_left = DetectFeatures();
        int num_coor_features = FindFeaturesInRight();
        if (num_coor_features < num_features_init_) {
            return false;
        }

        bool build_map_success = BuildInitMap();

        if (build_map_success) {
            status_ = FrontendStatus::TRACKING_GOOD;
            if (viewer_) {
                viewer_->AddCurrentFrame(current_frame_);
                viewer_->UpdateMap();
            }
            return true;
        }
        return false;

    }

    bool Frontend::Reset(){
        return true;
    }

    int Frontend::DetectFeatures(){
        // use gftt detector and append key point to current frame features  left
        cv::Mat mask(current_frame_->left_img_.size(), CV_8UC1, 255);
        /**
         * for each feature in the curent frame we draw a rectangle to represent the point
         * and render it on the mask template we just created
         *
         * the rectangle is the feature point offset by 10 x, y in corner
         * */


        for (auto &feat : current_frame_->features_left_) {
            cv::rectangle(mask, feat->position_.pt - cv::Point2f(10, 10),
                          feat->position_.pt + cv::Point2f(10, 10), 0, cv::FILLED);
        }

        std::vector<cv::KeyPoint> keypoints;
        // to detect keypoints in left image
        gftt_->detect(current_frame_->left_img_, keypoints, mask);
        int cnt_detected = 0;
        for (auto &kp : keypoints) {
            current_frame_->features_left_.push_back(
                    Feature::Ptr(new Feature(current_frame_, kp)));
            cnt_detected++;
        }

        LOG(INFO) << "Detect " << cnt_detected << " new features";
        return cnt_detected;

    }

    int Frontend::FindFeaturesInRight() {
        /// using LK flow to deterimne features in right image
        std::vector<cv::Point2f> kps_left, kps_right;

        for (auto &kp : current_frame_->features_left_){
            kps_left.push_back(kp->position_.pt);
            // to increase better chances we use 3d point of left key point to produce appoximation of points.
            //
            auto map_point = kp->map_point_.lock();
            if (map_point){
                auto pixel_pos = camera_right_->world2pixel(map_point->pos_, current_frame_->Pose());
                kps_right.push_back(cv::Point2f(pixel_pos[0], pixel_pos[1]));
            } else {
                // use same pixel in left iamge
                kps_right.push_back(kp->position_.pt);
            }
        }

        std::vector<uchar> status;
        Mat error;
        cv::calcOpticalFlowPyrLK(
                current_frame_->left_img_,\
                current_frame_->right_img_,
                kps_left,
                kps_right,
                status,
                error,
                cv::Size(11, 11),
                3,
                cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                                 0.01),
                cv::OPTFLOW_USE_INITIAL_FLOW);

        int num_good_pts = 0;
        for (size_t i = 0; i < status.size(); ++i) {
            if (status[i]) {
                cv::KeyPoint kp(kps_right[i], 7);
                Feature::Ptr feat(new Feature(current_frame_, kp));
                feat->is_on_left_image_ = false;
                current_frame_->features_right_.push_back(feat);
                num_good_pts++;
            } else {
                current_frame_->features_right_.push_back(nullptr);
            }
        }
        LOG(INFO) << "Find " << num_good_pts << " in the right image.";
        return num_good_pts;
        return 0;
    }

    bool Frontend::BuildInitMap(){
        return true;
    }


    bool Frontend::Track() {
        if (last_frame_){
            current_frame_->SetPose(relative_motion_ * last_frame_->Pose());
        }
        int num_track_last = TrackLastFrame();
        tracking_inliers_ = EstimateCurrentPose();

        if (tracking_inliers_ > num_features_tracking_ ){
            // number of inliners is in rightthreshold
            // set status
            status_ = FrontendStatus::TRACKING_GOOD;
        } else if(tracking_inliers_ > num_features_tracking_bad_){
            status_ = FrontendStatus::TRACKING_BAD;

        }else {
            status_ = FrontendStatus::LOST;
        }

        InsertKeyframe();
        // update key frame and update current pose
        relative_motion_ = current_frame_->Pose() * last_frame_->Pose().inverse();

//       if (viewer_) viewer_->AddCurrentFrame(current_frame_);
        return true;



    }

    int Frontend::TrackLastFrame() {
        // use LK flow to stimate points in the right image
        std::vector<cv::Point2f> kps_last, kps_current;
        //loop through features from last frame
        for (auto &kp : last_frame_->features_left_){
            if (kp->map_point_.lock()){
                auto mp = kp->map_point_.lock(); // get the map point from particular frame
                // pixel value of map point
                auto pixel = camera_left_->world2pixel(mp->pos_, current_frame_->Pose());
                kps_last.push_back(kp->position_.pt);
                kps_current.push_back(cv::Point2f( pixel[0], pixel[1] ));
            }
            else{
                kps_last.push_back(kp->position_.pt);
                kps_current.push_back(kp->position_.pt);
            }

        }
        std::vector<uchar> status;
        Mat error;
        cv::calcOpticalFlowPyrLK(last_frame_->left_img_,
                                 current_frame_->left_img_,
                                 kps_last,
                                 kps_current,
                                 status,
                                 error,
                                 cv::Size(21, 21),
                                 3,
                                 cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
                                 cv::OPTFLOW_USE_INITIAL_FLOW);
        int num_good_pts = 0;
        for (size_t i = 0; i < status.size(); ++i) {
            if (status[i]) {
                cv::KeyPoint kp(kps_current[i], 7);
                Feature::Ptr feature(new Feature(current_frame_, kp));
                feature->map_point_ = last_frame_->features_left_[i]->map_point_;
                current_frame_->features_left_.push_back(feature);
                num_good_pts++;
            }
        }
        return 0;
    }

    int Frontend::EstimateCurrentPose(){
        // we use g2o for this
        // implements bundle adjustment PnP on current frame
        typedef g2o::BlockSolver_6_3 BlockSolverType; // bA solver
        typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

        auto solver = new g2o::OptimizationAlgorithmLevenberg(
                g2o::make_unique<BlockSolverType>(
                        g2o::make_unique<LinearSolverType>()));
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(solver);

        // vertexs
        VertexPose *vertex_pose = new VertexPose();

        vertex_pose->setId(0);
        vertex_pose->setEstimate(current_frame_->Pose());

        optimizer.addVertex(vertex_pose);

        // camera intristics
        Mat33 k = camera_left_->K();


        // edges
        int idx = 1;
        // vector to store edges
        std::vector<EdgeProjectionPoseOnly *> edges;
        std::vector<Feature::Ptr> features;

        for (size_t i = 0; i< current_frame_->features_left_.size(); i++){
            auto map_point = current_frame_->features_left_[i]->map_point_.lock();
            //mappoint  is used to get 3d position probbaly with triangulation or 3d disparity maps
            if (map_point){
                features.push_back(current_frame_->features_left_[i]);
                EdgeProjectionPoseOnly *edge =
                        new EdgeProjectionPoseOnly(map_point->pos_, k);
                edge->setId(idx);
                edge->setVertex(0, vertex_pose);

                edge->setMeasurement(
                        toVec2(current_frame_->features_left_[i]->position_.pt));

                edge->setInformation(Eigen::Matrix2d::Identity());
                edge->setRobustKernel(new g2o::RobustKernelHuber);
                edges.push_back(edge);
                optimizer.addEdge(edge);
                idx++;
            }
        }

        // estimate the pose
        const double chi2_threshold = 5.991;
        int number_outlier = 0;
        for (int iter = 0; iter<4 ; iter++){
            vertex_pose->setEstimate(current_frame_->Pose());

            optimizer.initializeOptimization();
            optimizer.optimize(10);

            number_outlier = 0;

            // count the outliers
            for (size_t i = 0; i < edges.size(); ++i) {
                auto e = edges[i];
                if (features[i]->is_outlier_) {
                    e->computeError();
                }
                if (e->chi2() > chi2_threshold) {
                    features[i]->is_outlier_ = true;
                    e->setLevel(1);
                    number_outlier++;
                } else {
                    features[i]->is_outlier_ = false;
                    e->setLevel(0);
                };

                if (iter == 2) {
                    e->setRobustKernel(nullptr);
                }
            }
        }

        LOG(INFO) << "Outlier/Inlier in pose estimating: " << number_outlier << "/"
                  << features.size() - number_outlier;
        // Set pose and outlier
        current_frame_->SetPose(vertex_pose->estimate());

        LOG(INFO) << "Current Pose = \n" << current_frame_->Pose().matrix();

        for (auto &feat : features) {
            if (feat->is_outlier_) {
                feat->map_point_.reset();
                feat->is_outlier_ = false;  // maybe we can still use it in future
            }
        }
        return features.size() - number_outlier;
    }

    bool Frontend::InsertKeyframe(){
        if (tracking_inliers_ >= num_features_needed_for_keyframe_) {
            // still have enough features, don't insert keyframe
            return false;
        }

        // current frame is a new keyframe
        current_frame_->SetKeyFrame();
        map_->InsertKeyFrame(current_frame_);

        LOG(INFO) << "Set frame " << current_frame_->id_ << " as keyframe "
                  << current_frame_->key_frame_id_;


        SetObservationsForKeyFrame();
        DetectFeatures();  // detect new features

        // track in right image
        FindFeaturesInRight();
        // triangulate map points
        TriangulateNewPoints();
        // update backend because we have a new keyframe
        backend_->UpdateMap();

        if (viewer_) viewer_->UpdateMap();

        return true;
    }

    void Frontend::SetObservationsForKeyFrame() {
        // loop  through each features of current frame and map the map point back to the observation
        //Set the features in keyframe as new observation of the map points
        for (auto &feat : current_frame_->features_left_) {
            auto map_point = feat->map_point_.lock();
            if (map_point) {
                map_point->AddObservation(feat);
            }
        }
    }

    int Frontend:: TriangulateNewPoints(){
        return 0;
    }

//

//

}

