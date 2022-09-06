//
// Created by Tosin Oseni on 05/09/2022.
//

#ifndef STEREO_SLAM__ALGORITHM_H
#define STEREO_SLAM__ALGORITHM_H

#include "common_include.h"

namespace stereoSlam{
    inline Vec2 toVec2(const cv::Point2f p) {
        return Vec2(p.x, p.y);
    }
}
#endif //STEREO_SLAM__ALGORITHM_H
