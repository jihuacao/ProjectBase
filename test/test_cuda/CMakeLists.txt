# https://developer.nvidia.com/blog/building-cuda-applications-cmake/
cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
project(test_cuda LANGUAGES CXX CUDA)

set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CUDA_BUILD_CUBIN ON)
include(FindCUDA)
CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS 3.0 3.5+PTX 5.2 Maxwell)
message(STATUS ${ARCH_FLAGS})
#list(APPEND CUDA_NVCC_FLAGS "${ARCH_FLAGS} --profile")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")  # enable cuda-gdb
endif()
 
include(CTest)
 
#set(cuset culib.cu culib2.cu kernel.cu)
#CUDA_COMPILE_PTX(cuda_ptx_files ${cuset})
#add_custom_target(ptx ALL DEPENDS ${cuda_ptx_files} ${cuset} SOURCES ${cuset})
#CUDA_COMPILE_CUBIN(cuda_cubin_files ${cuset})
#add_custom_target(cubin ALL DEPENDS ${cuda_cubin_files} ${cuset} SOURCES ${cuset})
set(cuset kernel-ptx-cubin.cu)
CUDA_COMPILE_PTX(cuda_ptx_files ${cuset})
add_custom_target(ptx ALL DEPENDS ${cuda_ptx_files} ${cuset} SOURCES ${cuset})
CUDA_COMPILE_CUBIN(cuda_cubin_files ${cuset})
add_custom_target(cubin ALL DEPENDS ${cuda_cubin_files} ${cuset} SOURCES ${cuset})

# We need to explicitly state that we need all CUDA files in the 
# particle library to be built with -dc as the member functions 
# could be called by other libraries and executables
#set_target_properties(
#  culib
#  PROPERTIES
#    CUDA_SEPARABLE_COMPILATION ON
#    CUDA_PTX_COMPILATION ON
#)
 
add_executable(
  test_cuda
  kernel.cuh
  kernel.cu
  test_cuda.hpp
  test_cuda.cpp
)
 
set_property(
  TARGET test_cuda
  PROPERTY
    CUDA_SEPARABLE_COMPILATION ON
)
include_directories(${CUDA_INCLUDE_DIRS})
target_link_libraries(test_cuda PUBLIC ${CUDA_LIBRARIES})