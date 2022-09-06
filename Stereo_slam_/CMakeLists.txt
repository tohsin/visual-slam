cmake_minimum_required(VERSION 3.22)
project(Stereo_slam_)

set(CMAKE_CXX_STANDARD 14)

#add_executable(Stereo_slam_ main.cpp src/frontend.cpp)

find_package(OpenGL REQUIRED QUIET)
include_directories(${OPENGL_INCLUDE_DIR})




list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

find_package(fmt REQUIRED)


find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#Eigen
Find_Package(Eigen3 REQUIRED)
include_directories("/usr/include/eigen3")

# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# G2O
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})

set(THIRD_PARTY_LIBS
        OpenGL::GL
        Eigen3::Eigen
        ${OpenCV_LIBS}
         ${Sophus_LIBRARIES}
        ${OPENGL_LIBRARIES}
        ${Pangolin_LIBRARIES}
         ${FMT_LIBRARIES}
        ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY}
        ${G20_TYPES_LIBRARY}
        ${G2O_SOLVER_CSPARSE}
        ${GTEST_BOTH_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${GFLAGS_LIBRARIES}
        pthread
        ${CSPARSE_LIBRARY}
         Sophus::Sophus

        )

enable_testing()


############### source and test ######################
include_directories(${PROJECT_SOURCE_DIR}/include)
add_subdirectory(src)
#add_subdirectory(test)
#add_subdirectory(app)
