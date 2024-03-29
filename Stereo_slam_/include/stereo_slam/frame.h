//
// Created by Tosin Oseni on 26/08/2022.
//
#pragma once

#ifndef STEREO_SLAM__FRAME_H
#define STEREO_SLAM__FRAME_H


#include "stereo_slam/common_include.h"
#include "stereo_slam/feature.h"
#include "stereo_slam/camera.h"
#include "stereo_slam/mappoint.h"

namespace stereoSlam{
//    struct Feature;

    struct Frame{

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        // variable ptr to point to frame
        typedef std::shared_ptr<Frame> Ptr;
        unsigned long id_ = 0; // id of the frame
        unsigned long key_frame_id_ = 0; // id of the key frame
        bool is_key_frame = false;
        double time_stamp_  = 0.0;
        SE3 pose_; // Tcw pose from camera to world

        std::mutex pose_mutex_; // pose data mutex
        cv::Mat left_img_, right_img_; //stereo images left and right

        // features gotten from images no need to store data on the features
        std::vector<std::shared_ptr<Feature>> features_left_;
        std::vector<std::shared_ptr<Feature>> features_right_;


    public:
        Frame(){} // empty constructor

        Frame(long id, double time_stamp, SE3 &pose, const Mat &left, const Mat &right);

        SE3 Pose() {
            std::unique_lock<std::mutex> lck(pose_mutex_);
            return pose_;
        }

        void SetPose(SE3 pose){
            std::unique_lock<std::mutex> lck(pose_mutex_);
            pose_ = pose;
        }

        void SetKeyFrame();

        static std::shared_ptr<Frame> CreateFrame();

    };
};
#endif //STEREO_SLAM__FRAME_H
