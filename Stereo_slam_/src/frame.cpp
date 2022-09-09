//
// Created by Tosin Oseni on 09/09/2022.
//

#include "stereo_slam/frame.h"
namespace stereoSlam{
    Frame::Frame(long id, double time_stamp, SE3 &pose, const Mat &left, const Mat &right)
        : id_(id), time_stamp_(time_stamp), pose_(pose), left_img_(left), right_img_(right){}

    void Frame::SetKeyFrame() {

    }

    Frame::Ptr Frame::CreateFrame() {
        static long factory_id = 0;
        Frame::Ptr new_frame(new Frame);
        new_frame->id_ = factory_id++;
        return new_frame;
    }

}