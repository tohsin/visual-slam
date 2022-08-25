cmake_minimum_required(VERSION 3.22)
project(Bundle_adjustment_BAL_dataset)

set(CMAKE_CXX_STANDARD 14)
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)


Find_Package(G2O REQUIRED)
Find_Package(Eigen3 REQUIRED)
Find_Package(Ceres REQUIRED)
Find_Package(Sophus REQUIRED)
Find_Package(CSPARSE REQUIRED)

SET(G2O_LIBS g2o_csparse_extension g2o_stuff g2o_core cxsparse)

include_directories(${PROJECT_SOURCE_DIR} ${EIGEN3_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR})

add_executable(Bundle_adjustment_BAL_dataset main.cpp)

add_library(bal_common common.cpp common.h)
target_link_libraries(Bundle_adjustment_BAL_dataset ${G2O_LIBS} bal_common)

