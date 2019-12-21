cmake_minimum_required(VERSION 3.5.1 FATAL_ERROR)

if(POLICY CMP0042)
cmake_policy(SET CMP0042 NEW)
endif()

if(POLICY CMP0048)
cmake_policy(SET CMP0048 NEW)
endif()

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


get_filename_component(project_include_dir ${ProjectBase_SOURCE_DIR} PATH)
include_directories(${project_include_dir})
message("${INCLUDE_DIRECTORIES}")
set(CMAKE_CXX_FLAGS "-fvisibility=hidden -std=c++11 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -Wall -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -Wall -DNDEBUG")
set(CMAKE_INSTALL_PREFIX, ${ProjectBase_SOURCE_DIR}/install)

# glog depend
add_subdirectory(util)
add_subdirectory(multi_language)