#pragma once

#include "stereo_slam/common_include.h"
#include "stereo_slam/frame.h"
#include "stereo_slam/map.h"

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

public:
    Map::Ptr map_ = nullptr;
    Frame::Ptr current_frame_ = nullptr;
    
}
}