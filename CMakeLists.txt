cmake_minimum_required(VERSION 3.5.1)
project(ProjectBase VERSION 0.1.0 LANGUAGES C CXX)
get_filename_component(project_include_dir ${ProjectBase_SOURCE_DIR} PATH)
include_directories(${project_include_dir})
message("${INCLUDE_DIRECTORIES}")
set(CMAKE_CXX_FLAGS "-fvisibility=hidden -std=c++11 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -Wall -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -Wall -DNDEBUG")
set(CMAKE_INSTALL_PREFIX, ${ProjectBase_SOURCE_DIR}/install)

# glog depend
include(external_process)

add_subdirectory(util)
add_subdirectory(multi_language)