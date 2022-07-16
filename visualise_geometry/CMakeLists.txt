cmake_minimum_required(VERSION 3.22)
project(visualise_geometry)

set(CMAKE_CXX_STANDARD 14)

#include_directories( "/usr/include/eigen3" )

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package( Pangolin )
include_directories( ${Pangolin_INCLUDE_DIRS} )

add_executable(visualise_geometry main.cpp)
target_link_libraries( visualise_geometry ${Pangolin_LIBRARIES} )
target_link_libraries(visualise_geometry Eigen3::Eigen)