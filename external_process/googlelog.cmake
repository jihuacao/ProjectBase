# base on glog v2.2.2
message(${CMAKE_CURRENT_LIST_FILE})
include(popular_message)
project_base_system_message()

find_package(glog)

if(${glog_FOUND})
    message("${CMAKE_INCLUDE_PATH}")
    message("${CMAKE_FIND_ROOT_PATH}")
    message("${glog_INCLUDE_DIRS}")
    message("${glog_VERSION}")
else()
    include(ExternalProject)

    set(glog_url https://github.com/google/glog.git)

    list(APPEND all_version "0.4.0")
    version_selector(glog all_version "0.4.0")

    list(APPEND supported_version "0.4.0")
    list(APPEND supported_tag "v0.4.0")
    version_tag_matcher(glog supported_version supported_tag glog_version)

    default_external_project_build_type(glog)

    ## this is a list contain versions which have the same behavior
    ##set(all_version "0.4.0")
    ##set(uniform_version_one "0.4.0")
    ##set(glog_version "0.4.0" CACHE STRING "glog version")
    ##set_property(CACHE glog_version PROPERTY STRINGS ${all_version})
    #if(${glog_version} STREQUAL "1.8.x")
    #    set(glog_tag v1.8.x)
    #if(${glog_version} STREQUAL "1.10.x")
    #    set(glog_tag v1.10.x)
    #else()
    #    message(FATAL_ERROR "unsupported glog version: ${glog_version}")
    #endif()
    #
    ## to get the gfalgs_build_type
    #set(supported_glog_build_type "FOLLOW_CMAKE_BUILD_TYPE" "Release" "Debug")
    #set(glog_build_type "FOLLOW_CMAKE_BUILD_TYPE" CACHE STRING "the specifical option for glog, if the glog_build_type is set")
    #set_property(CACHE glog_build_type PROPERTY STRINGS "Release" "Debug" "FOLLOW_CMAKE_BUILD_TYPE")
    #if(glog_build_type STREQUAL "")
    #    set(glog_build_type ${CMAKE_BUILD_TYPE})
    #elseif(glog_build_type STREQUAL "FOLLOW_CMAKE_BUILD_TYPE")
    #    set(glog_build_type ${CMAKE_BUILD_TYPE})
    #elseif(glog_build_type STREQUAL "R")
    #elseif(${glog_build_type} IN_LIST supported_glog_build_type)
    #else()
    #    message(FATAL_ERROR "unsupported glog_build_type: ${glog_build_type}")
    #endif()
    #
    ## to get the _glog_build_type shared or static
    #set(glog_build_shared "FOLLOW_EXTERNAL_BUILD_SHARED" CACHE STRING "specifical build the glog in shared or follow in external_build_shared")
    #set_property(CACHE glog_build_shared PROPERTY STRINGS "ON" "OFF" "FOLLOW_EXTERNAL_BUILD_SHARED")
    #if(glog_build_shared STREQUAL "FOLLOW_EXTERNAL_BUILD_SHARED")
    #    set(_glog_build_shared ${external_build_shared})
    #elseif(glog_build_shared STREQUAL "")
    #    set(_glog_build_shared ${external_build_shared})
    #elseif(glog_build_shared STREQUAL "ON")
    #    set(_glog_build_shared ON)
    #elseif(glog_build_shared STREQUAL "OFF")
    #    set(_glog_build_shared OFF)
    #else()
    #    message(FATAL_ERROR "unsupported glog_build_shared: ${glog_build_shared}")
    #endif()
    #
    #message("glog: version->4{glog_version} build in->${glog_build_type} shared?->${_glog_build_shared}")
    #
    #if(${glog_version} IN_LIST uniform_version_one)
    #    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    #        if(glog_build_type STREQUAL "Debug")
    #            set(glog_lib_name glogd.lib)
    #            set(gmock_lib_name gmockd.lib)
    #        elseif(glog_build_type STREQUAL "Release")
    #            set(glog_lib_name glog.lib)
    #            set(gmock_lib_name gmock.lib)
    #        endif()
    #    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    #        if(glog_build_type STREQUAL "Debug")
    #            if(_glog_build_shared)
    #                set(glog_lib_name libglogd.so)
    #                set(gmock_lib_name libgmockd.so)
    #            else()
    #                set(glog_lib_name libglogd.a)
    #                set(gmock_lib_name libgmockd.a)
    #            endif()
    #        elseif(glog_build_type STREQUAL "Release")
    #            if(_glog_build_shared)
    #                set(glog_lib_name libglog.so)
    #                set(gmock_lib_name libgmock.so)
    #            else()
    #                set(glog_lib_name libglog.a)
    #                set(gmock_lib_name libgmock.a)
    #            endif()
    #        endif()
    #    endif()
    #else()
    #endif()
    #
    #set(glog_include ${external_install_path}/include)
    #set(glog_lib_dir ${external_install_path}/lib)
    #set(glog_lib ${glog_lib_dir}/${glog_lib_name})
    #
    #ExternalProject_Add(glog
    #    PREFIX glog
    #    GIT_REPOSITORY ${glog_url}
    #    GIT_TAG v1.10.x
    #    DOWNLOAD_DIR ${external_download_dir}
    #    BUILD_IN_SOURCE 0
    #    BUILD_BYPRODUCTS ${glog_lib_name} ${glog_nothreads_lib_name}
    #    INSTALL_COMMAND make install
    #    BUILD_COMMAND make -j 8
    #    CMAKE_CACHE_ARGS
    #        -DBUILD_GMOCK:BOOL=ON
    #        -DBUILD_SHARED_LIBS:BOOL=${_glog_build_shared}
    #        -DCMAKE_BUILD_TYPE:STRING=${glog_build_type}
    #        -DCMAKE_INSTALL_PREFIX:STRING=${CMAKE_SOURCE_DIR}/install
    #        -DINSTALL_GTEST:BOOL=ON
    #        -Dgmock_build_tests:BOOL=OFF
    #        -Dglog_build_samples:BOOL=OFF
    #        -Dglog_build_tests:BOOL=OFF
    #        -Dglog_disable_pthreads:BOOL=OFF
    #        -Dglog_force_shared_crt:BOOL=ON
    #        -Dglog_hide_internal_symbols:BOOL=OFF
    #)

endif()