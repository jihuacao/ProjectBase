# base on gflags v2.2.2
message(${CMAKE_CURRENT_LIST_FILE})
include(popular_message)
include(ExternalProject)
project_base_system_message()

set(gflags_build_type "" CACHE STRING "the specifical option for gflags, if the gflags_build_type is set")
set_property(CACHE gflags_build_type PROPERTY STRINGS "Release" "Debug")
if(gflags_build_type STREQUAL "")
    set(gflags_build_type ${CMAKE_BUILD_TYPE})
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    if(gflags_build_type STREQUAL "Debug")
        if(external_build_shared)
            set(gflags_lib_name gflags_debug.lib)
            set(gflags_nothreads_lib_name gflags_nothreads_debug.lib)
        else()
            set(gflags_lib_name gflags_static_debug.lib)
            set(gflags_nothreads_lib_name gflags_nothread_static_debug.lib)
        endif()
    elseif(gflags_build_type STREQUAL "Release")
        if(external_build_shared)
            set(gflags_lib_name gflags.lib)
            set(gflags_nothreads_lib_name gflags_nothreads.lib)
        else()
            set(gflags_lib_name gflags_static.lib)
            set(gflags_nothreads_lib_name gflags_nothreads_static.lib)
        endif()
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(gflags_build_type STREQUAL "Debug")
        if(external_build_shared)
            set(gflags_lib_name libflags_debug.so)
            set(gflags_nothreads_lib_name libflags_nothreads_debug.so)
        else()
            set(gflags_lib_name libflags_debug.a)
            set(gflags_nothreads_lib_name libflags_nothreads_debug.a)
        endif()
    elseif(gflags_build_type STREQUAL "Release")
        if(external_build_shared)
            set(gflags_lib_name libflags.so)
            set(gflags_nothreads_lib_name libflags_nothreads.so)
        else()
            set(gflags_lib_name libflags.a)
            set(gflags_nothreads_lib_name libflags_nothreads.a)
        endif()
    endif()
endif()

set(gflags_url https://github.com/gflags/gflags.git)
set(gflags_version "2.2.2" CACHE STRING "gflags version")
set_property(CACHE gflags_version PROPERTY STRINGS "2.2.2" "2.2.1")
if(${gflags_version} STREQUAL "2.2.2")
    set(gflags_tag v2.2.2)
elseif(${gflags_version} STREQUAL "2.2.1")
    set(gflags_tag v2.2.2)
endif()
set(gflags_lib ${external_install_path}/lib/${gflags_lib_name})

ExternalProject_Add(gflags
    PREFIX gflags
    GIT_REPOSITORY ${gflags_url}
    GIT_TAG ${gflags_tag}
    DOWNLOAD_DIR ${external_download_dir}
    BUILD_IN_SOURCE 1
    BUILD_BYPRODUCTS ${gflags_lib_name} ${gflags_nothreads_lib_name}
    INSTALL_COMMAND ""
    CMAKE_CACHE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${gflags_build_type}
        -DBUILD_PACKAGING:BOOL=OFF
        -DBUILD_SHARED_LIBS:BOOL=${external_build_shared}
        -DBUILD_TESTING:BOOL=OFF
        -DBUILD_gflags_LIB:BOOL=ON
        -DBUILD_gflags_nothreads_LIB:BOOL=ON
        -DREGISTER_BUILD_DIR:BOOL=OFF
        -DREGISTER_INSTALL_PREFIX:BOOL=OFF
)