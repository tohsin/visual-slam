cmake_minimum_required(VERSION 3.22)
project(image_basics)

find_package(OpenCVREQUIRED)
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 20)

add_executable(image_basics main.cpp)

target_link_libraries(image_basics ${OpenCV_LIBS})