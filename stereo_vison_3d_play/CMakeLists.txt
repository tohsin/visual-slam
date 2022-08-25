cmake_minimum_required(VERSION 3.22)
project(stereo_vison_3d_play)
set(Pangolin_INCLUDE_DIRS /usr/local/include)

find_package(OpenCV REQUIRED)
find_package(OpenCV)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package( Pangolin )
include_directories( ${Pangolin_INCLUDE_DIRS} )

include_directories(${OpenCV_INCLUDE_DIRS})


set(CMAKE_CXX_STANDARD 14)

add_executable(stereo_vison_3d_play main.cpp)
target_link_libraries(stereo_vison_3d_play ${OpenCV_LIBS} ,  Eigen3::Eigen)
target_link_libraries( stereo_vison_3d_play ${Pangolin_LIBRARIES} )


