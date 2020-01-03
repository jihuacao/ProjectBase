message(STATUS ${CMAKE_CURRENT_LIST_FILE})

set(with_Boost ON)
include(popular_message)
project_base_system_message()
include(ExternalProject)

include(external_setting)

set(module Boost)
list(
    APPEND 
    ${module}_all_version 
    "1.71.0"
    )

set(
    ${module}_supported_url 
    https://dl.bintray.com/Boostorg/release/1.71.0/source/Boost_1_71_0.tar.gz
    )

set(
    ${module}_supported_hash
    96b34f7468f26a141f6020efb813f1a2f3dfb9797ecf76a7d7cbd843cc95f5bd
    )

version_selector(${module} ${module}_all_version "1.71.0")

# to get the Boost_build_type
default_external_project_build_type(${module})

# to get the Boost_build_shared
project_build_shared(${module})

# the BoostConfig.cmake config the Boost_INCLUDE_DIRS for include, 
#
# set(Boost_USE_DEBUG_LIBS ON)
# set(${module}_USE_STATIC_LIBS OFF)
# set(Boost_DEBUG ON)
find_package(${module} 1.71 REQUIRED COMPONENTS filesystem CONFIG HINTS /home/sins/Download/boost_1_71_0/install/)
message("libs:${${module}_LIBRARIES}")
include_directories(${${module}_INCLUDE_DIRS})
link_directories(${${module}_DIR}/lib)

get_cmake_property(_variableNames VARIABLES)
# message("sd:${_variableNames}")

if(${CMAKE_BUILD_TYPE} STREQUAL "Release")
else()
endif()
if(${module}_FOUND)
else()
    version_url_hash_match(${module} ${module}_all_version ${module}_supported_url ${module}_supported_hash ${module}_version)

    # mkdir the 
    ExternalProject_Add(
        Boost
        PREFIX Boost-${${module}_version}
        URL ${${module}_url}
        URL_HASH:SHA256="${${module}_hash}"
        DOWNLOAD_COMMAND axel -k -n 10 -av ${${module}_url}
        DOWNLOAD_DIR "${external_download_dir}"

        UPDATE_COMMAND ""
    )
endif()