# base on gflags v2.2.2
message(${CMAKE_CURRENT_LIST_FILE})
include(popular_message)
include(ExternalProject)
project_base_system_message()

set(gflags_url https://github.com/gflags/gflags.git)

# this is a list contain versions which have the same behavior
set(all_version "2.2.2")
set(uniform_version_one "2.2.2")
set(gflags_version "2.2.2" CACHE STRING "gflags version")
set_property(CACHE gflags_version PROPERTY STRINGS ${all_version})
if(${gflags_version} STREQUAL "2.2.2")
    set(gflags_tag v2.2.2)
else()
    message(FATAL_ERROR "unsupported gfalgs version: ${gflags_version}")
endif()

# to get the gfalgs_build_type
set(supported_gflags_build_type "FOLLOW_CMAKE_BUILD_TYPE" "Release" "Debug")
set(gflags_build_type "FOLLOW_CMAKE_BUILD_TYPE" CACHE STRING "the specifical option for gflags, if the gflags_build_type is set")
set_property(CACHE gflags_build_type PROPERTY STRINGS "Release" "Debug" "FOLLOW_CMAKE_BUILD_TYPE")
if(gflags_build_type STREQUAL "")
    set(gflags_build_type ${CMAKE_BUILD_TYPE})
elseif(gflags_build_type STREQUAL "FOLLOW_CMAKE_BUILD_TYPE")
    set(gflags_build_type ${CMAKE_BUILD_TYPE})
elseif(gflags_build_type STREQUAL "R")
elseif(${gflags_build_type} IN_LIST supported_gflags_build_type)
else()
    message(FATAL_ERROR "unsupported gflags_build_type: ${gflags_build_type}")
endif()

# to get the _gflags_build_type shared or static
set(gflags_build_shared "FOLLOW_EXTERNAL_BUILD_SHARED" CACHE STRING "specifical build the gflags in shared or follow in external_build_shared")
set_property(CACHE gflags_build_shared PROPERTY STRINGS "ON" "OFF" "FOLLOW_EXTERNAL_BUILD_SHARED")
if(gflags_build_shared STREQUAL "FOLLOW_EXTERNAL_BUILD_SHARED")
    set(_gflags_build_shared ${external_build_shared})
elseif(gflags_build_shared STREQUAL "")
    set(_gflags_build_shared ${external_build_shared})
elseif(gflags_build_shared STREQUAL "ON")
    set(_gflags_build_shared ON)
elseif(gflags_build_shared STREQUAL "OFF")
    set(_gflags_build_shared OFF)
else()
    message(FATAL_ERROR "unsupported gflags_build_shared: ${gflags_build_shared}")
endif()

message("gflags: version->4{gflags_version} build in->${gflags_build_type} shared?->${_gflags_build_shared}")

if(${gflags_version} IN_LIST uniform_version_one)
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        if(gflags_build_type STREQUAL "Debug")
            if(_gflags_build_shared)
                set(gflags_lib_name gflags_debug.lib)
                set(gflags_nothreads_lib_name gflags_nothreads_debug.lib)
            else()
                set(gflags_lib_name gflags_static_debug.lib)
                set(gflags_nothreads_lib_name gflags_nothread_static_debug.lib)
            endif()
        elseif(gflags_build_type STREQUAL "Release")
            if(_gflags_build_shared)
                set(gflags_lib_name gflags.lib)
                set(gflags_nothreads_lib_name gflags_nothreads.lib)
            else()
                set(gflags_lib_name gflags_static.lib)
                set(gflags_nothreads_lib_name gflags_nothreads_static.lib)
            endif()
        endif()
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        if(gflags_build_type STREQUAL "Debug")
            if(_gflags_build_shared)
                set(gflags_lib_name libflags_debug.so)
                set(gflags_nothreads_lib_name libflags_nothreads_debug.so)
            else()
                set(gflags_lib_name libflags_debug.a)
                set(gflags_nothreads_lib_name libflags_nothreads_debug.a)
            endif()
        elseif(gflags_build_type STREQUAL "Release")
            if(_gflags_build_shared)
                set(gflags_lib_name libflags.so)
                set(gflags_nothreads_lib_name libflags_nothreads.so)
            else()
                set(gflags_lib_name libflags.a)
                set(gflags_nothreads_lib_name libflags_nothreads.a)
            endif()
        endif()
    endif()
else()
endif()

set(gflags_include ${external_install_path}/include)
set(gflags_lib_dir ${external_install_path}/lib)
set(gflags_lib ${gflags_lib_dir}/${gflags_lib_name})

ExternalProject_Add(gflags
    PREFIX gflags
    GIT_REPOSITORY ${gflags_url}
    GIT_TAG ${gflags_tag}
    DOWNLOAD_DIR ${external_download_dir}
    BUILD_IN_SOURCE 0
    BUILD_BYPRODUCTS ${gflags_lib_name} ${gflags_nothreads_lib_name}
    INSTALL_COMMAND make install
    BUILD_COMMAND make -j 8
    CMAKE_CACHE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${gflags_build_type}
        -DCMAKE_INSTALL_PREFIX:STRING=${CMAKE_SOURCE_DIR}/install
        -DBUILD_PACKAGING:BOOL=OFF
        -DBUILD_SHARED_LIBS:BOOL=${_gflags_build_shared}
        -DBUILD_TESTING:BOOL=OFF
        -DBUILD_gflags_LIB:BOOL=ON
        -DBUILD_gflags_nothreads_LIB:BOOL=ON
        -DREGISTER_BUILD_DIR:BOOL=OFF
        -DREGISTER_INSTALL_PREFIX:BOOL=OFF
)