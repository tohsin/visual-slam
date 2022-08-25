cmake_minimum_required(VERSION 3.22)
project(poseGraph_g2o)

set(CMAKE_CXX_STANDARD 14)
set(Sophus_INCLUDE_DIRS /usr/local/Cellar)
set(CMAKE_CXX_STANDARD 14)

Find_Package(Sophus REQUIRED)

include_directories(
            ${Sophus_INCLUDE_DIRS}
            "/usr/include/eigen3"
)
add_library(bal_common common.cpp )
target_link_libraries(bal_common Eigen3::Eigen)
add_executable(poseGraph_g2o main.cpp)
target_link_libraries(poseGraph_g2o Eigen3::Eigen)
target_link_libraries(poseGraph_g2o Sophus::Sophus ${Sophus_LIBRARIES}  bal_common)
