include(add_cu_module)
add_cu_base_module(Yolo)
#enable_language(CUDA)
#
#add_definitions(-std=c++11)
#option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
#set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_CUDA_FLAGS "--keep --generate-code arch=compute_62,code=compute_62 --generate-code arch=compute_72,code=compute_72 ${CMAKE_CUDA_FLAGS}")
#
## TODO(Call for PR): make cmake compatible with Windows
#
## include and link dirs of cuda and tensorrt, you need adapt them if yours are different
## cuda
#
#include_directories(/usr/include/x86_64-linux-gnu)
#link_directories(/usr/lib/x86_64-linux-gnu)
#
##file(GLOB_RECURSE PLUGIN_SRCS ${PROJECT_SOURCE_DIR}/*.cu ${PROJECT_SOURCE_DIR}/src/*.cc ${PROJECT_SOURCE_DIR}/src/*.cpp)
#message(STATUS ${CMAKE_CURRENT_SOURCE_DIR})
##file(GLOB PLUGIN_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cc ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
#
#aux_source_directory(${CMAKE_CURRENT_LIST_DIR} PLUGIN_SRCS)
#set(PLUGIN_SRCS ${CMAKE_CURRENT_LIST_DIR}/yolov5.cu ${CMAKE_CURRENT_LIST_DIR}/yolov5.cpp)
#message(STATUS ${PLUGIN_SRCS})
#
#add_library(Yolo SHARED ${PLUGIN_SRCS})
#target_link_libraries(myplugins nvinfer cudart)
#add_subdirectory(yolo_test)