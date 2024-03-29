//
// Created by Tosin Oseni on 26/08/2022.
//
#pragma once
#ifndef STEREO_SLAM__MAP_H
#define STEREO_SLAM__MAP_H



#include "stereo_slam/common_include.h"
#include "stereo_slam/mappoint.h"
#include "stereo_slam/frame.h"
namespace stereoSlam{

    struct Map{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Map> Ptr;
        // map is just a collection of map points
        typedef std::unordered_map<unsigned long, MapPoint::Ptr> LandmarksType;
        typedef std::unordered_map<unsigned long, Frame::Ptr> KeyframesType;

        Map() {}

        void InsertKeyFrame(Frame::Ptr frame);

        void InsertMapPoint(MapPoint::Ptr map_point);

        LandmarksType GetAllMapPoints() {
            std::unique_lock<std::mutex> lck(data_mutex_);
            return landmarks_;
        }

        KeyframesType GetAllKeyFrames() {
            std::unique_lock<std::mutex> lck(data_mutex_);
            return keyframes_;
        }

        LandmarksType GetActiveMapPoints() {
            std::unique_lock<std::mutex> lck(data_mutex_);
            return active_landmarks_;
        }

        KeyframesType GetActiveKeyFrames() {
            std::unique_lock<std::mutex> lck(data_mutex_);
            return active_keyframes_;
        }

    private:
        void CleanMap();
        void RemoveOldKeyframe();


        std::mutex data_mutex_;
        LandmarksType landmarks_;         // all landmarks
        LandmarksType active_landmarks_;  // active landmarks
        KeyframesType keyframes_;         // all key-frames
        KeyframesType active_keyframes_;  // active key-frames

        Frame::Ptr current_frame_ = nullptr;

        int num_active_keyframes_ = 7;

    };
}

// n short, the frame holds the shared_ptr of feature,
//so we should avoid the feature holding frame’s shared_ptr again. Otherwise, the two structs refer to each other,
//which will cause the smart pointer to fail to be automatically destructed.
#endif //STEREO_SLAM__MAP_H
