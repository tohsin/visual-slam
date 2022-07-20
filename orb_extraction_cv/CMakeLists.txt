cmake_minimum_required(VERSION 3.22)
project(orb_extraction_cv)

find_package(OpenCV REQUIRED)
find_package(OpenCV)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)


include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 14)

add_executable(orb_extraction_cv main.cpp)
target_link_libraries(orb_extraction_cv  ${OpenCV_LIBS})
