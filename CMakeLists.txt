cmake_minimum_required(VERSION 3.19.0 FATAL_ERROR)

if(POLICY CMP0042)
cmake_policy(SET CMP0042 NEW)
endif()

if(POLICY CMP0048)
cmake_policy(SET CMP0048 NEW)
endif()

execute_process(COMMAND uname -a
		COMMAND awk "{if ($0 ~ \"x86_64\") {print \"x64\"} else {print \"x86\"}}"
		OUTPUT_VARIABLE UNAME_LOG)
MESSAGE(STATUS "Your system architecture is: ${UNAME_LOG}" )

#----------------------------------------
# package information
set(PACKAGE_NAME "ProjectBase")
set(PACKAGE_VERSION "0.1.0")
set(PACKAGE_STRING "${PACKAGE_NAME} ${PACKAGE_VERSION}")
set(PACKAGE_TARNAME "${PACKAGE_NAME}-${PACKAGE_VERSION}")
set(PACKAGE_BUGREPORT "")
set(PACKAGE_DESCRIPTION "")
set(PACKAGE_URL "")
project(${PACKAGE_NAME} VERSION ${PACKAGE_VERSION} LANGUAGES C CXX)

# add cmake_module to CMAKE_MODULE_PATH
set(cmake_module_dir ${ProjectBase_SOURCE_DIR}/cmake_module)
set(CMAKE_MODULE_PATH ${cmake_module_dir} ${CMAKE_MODULE_PATH})
set(external_process ${ProjectBase_SOURCE_DIR}/external_process)
set(CMAKE_MODULE_PATH ${external_process} ${CMAKE_MODULE_PATH})
include(popular_message)
project_base_system_message()

#生成链接库
set(cm_module_name cm)
if (OPENCL)
    find_package(OpenCL REQUIRED)
    if (OpenCL_FOUND)
        message(STATUS "found opencl")
        set(${cm_module_name}_target_name opencl)
        add_library(${${cm_module_name}_target_name} INTERFACE IMPORTED)
        set_property(TARGET ${${cm_module_name}_target_name} PROPERTY
        	INTERFACE_INCLUDE_DIRECTORIES ${OpenCL_INCLUDE_DIRS})
        set_property(TARGET ${${cm_module_name}_target_name} PROPERTY
            INTERFACE_LINK_LIBRARIES ${OpenCL_LIBRARY})
        list (APPEND DARKNET_DEFINITIONS GPU CL_TARGET_OPENCL_VERSION=120)
        set(CMAKE_CXX_FLAGS "-DCOMPUTE_MACHINE -DOPENCL")
    else()
		message(FATAL "could not found opencl")
        add_library(${${cm_module_name}_target_name} INTERFACE IMPORTED)
    endif()
else()
if (CUDA)
endif()
endif()

if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
set(CMAKE_CXX_FLAGS "-fvisibility=hidden -std=c++11 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -Wall -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -Wall -DNDEBUG")
else()
endif()

get_filename_component(project_include_dir ${ProjectBase_SOURCE_DIR} PATH)
include_directories(${project_include_dir})

option(project_base_build_shared_lib "build the shared library or not" ON)
option(project_base_build_python_interface "build the interface for python or not" OFF)
option(project_base_build_with_opencl "build the project with opencl, if not the component which depend on opencl would not be built" ON)

# gflags depend
include(gflags)
# glog depend
include(glog)
# gtest depend
include(GTest)
# boost depend
include(boost)
# eigen3 depend
include(eigen3)
# protobuf
include(protobuf)
# opencl
include(opencl)

# glog depend

# build plugin_manager
add_subdirectory(plugin_manager)
# build cmake_module
add_subdirectory(cmake_module)
# build_external_process
add_subdirectory(external_process)
# build script
add_subdirectory(script)
# build config
add_subdirectory(config)
# build cross_platform
add_subdirectory(cross_platform)
# build util
add_subdirectory(util)
# build method
add_subdirectory(method)
# build multi_language
add_subdirectory(multi_language)
# build unified_type
add_subdirectory(unified_type)
# build tensor
add_subdirectory(tensor)
# build matrix
add_subdirectory(matrix)
# build data
add_subdirectory(data)
# build function
add_subdirectory(function)
# build function_solution
add_subdirectory(function_solution)
# build function_ui
add_subdirectory(function_ui)
# build ui_style_config
add_subdirectory(ui_style_config)
# build codec
add_subdirectory(codec)
# build play_with_now_code
add_subdirectory(play_with_now_code)
# build indicator
add_subdirectory(indicator)
# build number_theory
add_subdirectory(number_theory)
# build basic_calculation
add_subdirectory(basic_calculation)
# build basic_algorithm
add_subdirectory(basic_algorithm)
# build algorithm
add_subdirectory(algorithm)

include(test_engineer_config)
if(${build_test})
	message(STATUS "build test")
    add_subdirectory(test)
endif()