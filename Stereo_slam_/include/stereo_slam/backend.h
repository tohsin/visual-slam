//
// Created by Tosin Oseni on 06/09/2022.
//

#ifndef STEREO_SLAM__BACKEND_H
#define STEREO_SLAM__BACKEND_H

#include "stereo_slam/common_include.h"
#include "stereo_slam/frame.h"
#include "stereo_slam/map.h"
namespace stereoSlam {
    class Backend{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Backend> Ptr;
        void SetCameras(Camera::Ptr left, Camera::Ptr right) {
            cam_left_ = left;
            cam_right_ = right;
        }


        /// start backend thread in constructor
        Backend();

        void setMap(std::shared_ptr<Map> map ){
            map_ = map;
        }

        /**
         * update map and optimise
         */
        void UpdateMap();
        /**
         * stop running backend
         */
        void Stop();
    private:
        // backend thread
        void BackendLoop();

        // optimise the sliding window Bundle adjustment
        void Optimize(Map::KeyframesType& keyframes, Map::LandmarksType& landmarks);


        std::shared_ptr<Map> map_;
        std::thread backend_thread_;
        std::mutex data_mutex_;

        std::condition_variable map_update_;
        std::atomic<bool> backend_running_;
        Camera::Ptr cam_left_ = nullptr, cam_right_ = nullptr;
    };

};
#endif //STEREO_SLAM__BACKEND_H
