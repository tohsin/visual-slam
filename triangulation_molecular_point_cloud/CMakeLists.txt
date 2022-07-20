cmake_minimum_required(VERSION 3.22)
project(triangulation_molecular_point_cloud)
find_package(OpenCV REQUIRED)
find_package(OpenCV)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
set(CMAKE_CXX_STANDARD 14)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(triangulation_molecular_point_cloud main.cpp)
target_link_libraries(triangulation_molecular_point_cloud  ${OpenCV_LIBS})



