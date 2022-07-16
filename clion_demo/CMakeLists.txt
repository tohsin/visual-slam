cmake_minimum_required(VERSION 3.22)
project(clion_demo)

set(CMAKE_CXX_STANDARD 14)
#find_library(OpenCV)
find_package(OpenCV)

include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(clion_demo main.cpp)

target_link_libraries(clion_demo ${OpenCV_LIBS})
