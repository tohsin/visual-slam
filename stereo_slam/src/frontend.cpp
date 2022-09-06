
#include <opencv2/opencv.hpp>
#include "stereo_slam/frontend.h"

bool Frontend::AddFrame(stereoSlam::Frame::ptr frame){
    // impliment adding frame 
    current_frame_ = frame;
    switch (status_)
    {
    case /* constant-expression */:
        /* code */
        break;
    
    default:
        break;
    }
}