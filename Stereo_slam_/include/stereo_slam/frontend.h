//
// Created by Tosin Oseni on 26/08/2022.
//
#pragma once
#ifndef STEREO_SLAM__FRONTEND_H
#define STEREO_SLAM__FRONTEND_H




#include "stereo_slam/common_include.h"
#include "stereo_slam/frame.h"
#include "stereo_slam/map.h"
#include "stereo_slam/camera.h"

namespace stereoSlam{

    class Backend;
    class Viewer;

    enum class FrontendStatus { INITING, TRACKING_GOOD, TRACKING_BAD, LOST };

    class Frontend{

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Frontend> Ptr;

        Frontend();

        //External interface, add a frame and calculate its positioning result
        bool AddFrame(Frame::Ptr frame);


        void SetMap(Map::Ptr map) {
            // set pointer to map object
            map_ = map;
        }
    private:
        FrontendStatus status_ = FrontendStatus::INITING;

        bool Track();

        bool SteroInit();

        bool Reset();

        /**
        * Track with last frame
        * @return num of tracked points
        */

        int TrackLastFrame();

        /**
         * estimate current frame's pose
         * @return num of inliers
         */
        int EstimateCurrentPose();

        /**
        * set current frame as a keyframe and insert it into backend
        * @return true if success
        */
        bool InsertKeyframe();

        /**
        * Set the features in keyframe as new observation of the map points
        */
        void SetObservationsForKeyFrame();

        /**
         * Detect features in left image in current_frame_
         * keypoints will be saved in current_frame_
         * @return
         */
        int DetectFeatures();

        /**
        * Find the corresponding features in right image of current_frame_
        * @return num of features found
        */
        int FindFeaturesInRight();

        /**
        * Triangulate the 2D points in current frame
        * @return num of triangulated points
        */
        int TriangulateNewPoints();

        int tracking_inliers_ = 0;  // inliers, used for testing new keyframes
        int num_features_tracking_ = 50;
        int num_features_needed_for_keyframe_ = 80;

        Camera::Ptr camera_left_ = nullptr;
        Camera::Ptr camera_RIGHT_ = nullptr;
        Map::Ptr map_ = nullptr;
        Frame::Ptr current_frame_ = nullptr;

        Frame::Ptr last_frame_ = nullptr;
        std::shared_ptr<Viewer> viewer_ = nullptr;
        std::shared_ptr<Backend> backend_ = nullptr;



//        The relative motion between the current frame and the previous frame,
//        which is used to estimate the initial pose value of the current frame
        SE3 relative_motion_;


    };
}
#endif //STEREO_SLAM__FRONTEND_H
