add_library(Stereo_slam_ SHARED
        frontend.cpp ../include/stereo_slam/common_include.h ../include/stereo_slam/frontend.h ../include/stereo_slam/frame.h ../include/stereo_slam/map.h ../include/stereo_slam/mappoint.h ../include/stereo_slam/feature.h ../include/stereo_slam/camera.h camera.cpp ../include/stereo_slam/g2o_types.h ../include/stereo_slam/algorithm.h ../include/stereo_slam/backend.h)


target_link_libraries(Stereo_slam_
        ${THIRD_PARTY_LIBS}
        )
