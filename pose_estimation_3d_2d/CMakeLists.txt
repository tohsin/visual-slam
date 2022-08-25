cmake_minimum_required(VERSION 3.22)
project(pose_estimation_3d_2d)
set(G2O_INCLUDE_DIRS /usr/local/include)
set(Sophus_INCLUDE_DIRS /usr/local/Cellar)

find_package(OpenCV REQUIRED)
find_package(OpenCV)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(G2O REQUIRED)
find_package(Sophus REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS})

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        "/usr/include/eigen3/"
)
#include_directories(${g2o_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 14)

add_executable(pose_estimation_3d_2d main.cpp)

target_link_libraries(pose_estimation_3d_2d Eigen3::Eigen)
target_link_libraries( pose_estimation_3d_2d
        ${G2O_LIBS}
        ${OpenCV_LIBS})
