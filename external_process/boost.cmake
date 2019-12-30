message(${CMAKE_CURRENT_LIST_FILE})

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

get_property(out CACHE PROPERTY ${module}_version FULL_DOCS)
message(STATUS ${out})

# the BoostConfig.cmake config the Boost_INCLUDE_DIRS for include, 
#
set(Boost_USE_DEBUG_LIBS ON)
set(Boost_DEBUG ON)
find_package(${module} 1.71 REQUIRED COMPONENTS python CONFIG HINTS /home/sins/Download/boost_1_71_0/install/)
message("${${module}_DIR}")
message("${${module}_INCLUDE_DIRS}")
message("${${module}_LIBRARY_DIRS}")
include_directories(${${module}_INCLUDE_DIRS})
include_directories(${${module}_INCLUDE_DIRS})
get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()
message(STATUS ${o})
get_directory_property(output INCLUDE_DIRECTORIES) 
message(STATUS ${INCLUDE_DIRECTORIES})
message(STATUS ${output})
if(${CMAKE_BUILD_TYPE} STREQUAL "Release")
else()
endif()
if(${module}_FOUND)
else()
    version_url_hash_match(${module} ${module}_all_version ${module}_supported_url ${module}_supported_hash ${module}_version)

    # to get the gfalgs_build_type
    default_external_project_build_type(${module})

    project_build_shared(${module})

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