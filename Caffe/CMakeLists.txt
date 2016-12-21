cmake_minimum_required(VERSION 3.6)
project(CaffeModel)

set(CMAKE_CXX_STANDARD 11)

find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})

set(SOURCE_FILES main.cpp featureExtraction.cpp featureExtraction.h)
add_executable(CaffeModel ${SOURCE_FILES})
target_link_libraries(CaffeModel ${Caffe_LIBRARIES})