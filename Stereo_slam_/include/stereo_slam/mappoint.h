//
// Created by Tosin Oseni on 26/08/2022.
//


// to store map point or landmark
// map point can be pbserved by multiple views  or features or co vosibility graph
#pragma once


#ifndef STEREO_SLAM__MAPPOINT_H
#define STEREO_SLAM__MAPPOINT_H

#include "stereo_slam/common_include.h"

namespace stereoSlam{

    struct MapPoint{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Feature> Ptr;
        unsigned long id_ = 0;
        bool is_outlier = false;
        Vec3 pos_ = Vec3::Zero(); // 3d position
        std::mutex data_mutex_;
        int number_featured_times_ = 0;
        // observation_ variable records the features that observed this map point
        std::list<std::weak_ptr<Feature>> observations_;


        MapPoint();
        MapPoint(long id, Vec3 position);

        Vec3 Pose(){
            // getter for position
            std::unique_lock<std::mutex> lck(data_mutex_);
            return pos_;
        }

        void SetPose(const Vec3 &pos){
            std::unique_lock<std::mutex> lck(data_mutex_);
            pos_ = pos;
        }


        void AddObservation(std::shared_ptr<Feature> feature){
            //Because the feature may be judged as an outlier, it needs to be locked when the observation part is changed.
            std::unique_lock<std::mutex> lck(data_mutex_);
            observations_.push_back(feature);
            number_featured_times_++;
        }

        void RemoveObservation(std::shared_ptr<Feature> feat);

        // factory function
        static MapPoint::Ptr CreateNewMappoint();

        std::list<std::weak_ptr<Feature>> GetObs() {
            std::unique_lock<std::mutex> lck(data_mutex_);
            return observations_;
        }


    };

}
#endif //STEREO_SLAM__MAPPOINT_H
