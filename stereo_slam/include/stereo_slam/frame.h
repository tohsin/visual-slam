#pragma once

#include "stereo_slam/common_include.h"

namespace stereoSlam{
struct Frame{

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        // variable ptr to point to frame
        typedef std::shared_ptr<Frame> Ptr;
        unsigned long id_ = 0; // id of the frame
        unsigned long key_frame_id_ = 0; // id of the key frame
        bool is_key_frame = false;
        double timestamp  = 0.0;
        cv::Mat pose; // Tcw pose from camera to world

        std::mutex pose_mutex_; // pose data mutex
        cv::Mat left_image_, right_image_; //stereo images
}
}
