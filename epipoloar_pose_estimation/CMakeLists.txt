cmake_minimum_required(VERSION 3.22)
project(epipoloar_pose_estimation)

find_package(OpenCV REQUIRED)
find_package(OpenCV)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 14)

add_executable(epipoloar_pose_estimation main.cpp)
target_link_libraries(epipoloar_pose_estimation  ${OpenCV_LIBS})
