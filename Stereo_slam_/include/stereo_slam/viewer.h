//
// Created by Tosin Oseni on 08/09/2022.
//

#ifndef STEREO_SLAM__VIEWER_H
#define STEREO_SLAM__VIEWER_H
#include "stereo_slam/common_include.h"
#include <pangolin/pangolin.h>
#include "stereo_slam/frame.h"
#include "stereo_slam/map.h"
namespace stereoSlam{
class Viewer {
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Viewer> Ptr;
    void UpdateMap();
    void AddCurrentFrame(Frame::Ptr current_frame);

private:

};
}


#endif //STEREO_SLAM__VIEWER_H
