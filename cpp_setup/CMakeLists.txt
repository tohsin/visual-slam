cmake_minimum_required(VERSION 2.8)

project(HelloSLAM)

#include main file to be compiled

add_executable(helloSLAM main.cpp)

# to add library files
add_library(hello libHelloSLAM.cpp)


add_executable(useHello useHello.cpp)

target_link_libraries(useHello hello_shared)