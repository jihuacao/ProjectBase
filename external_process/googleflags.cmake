# base on gflags v2.2.2
include(popular_message)
include(ExternalProject)
include(external_setting)
cmakelists_base_header()
set(module gflags)
project_base_system_message()

set(gflags_url https://github.com/gflags/gflags.git)

# this is a list contain versions which have the same behavior
set(${module}_supported_version "2.2.2")
set(${module}_supported_tag "v2.2.2")
version_selector(${module} ${module}_supported_version "2.2.2")
message(STATUS ${${module}_version})
version_tag_matcher(${module} ${module}_supported_version ${module}_supported_tag ${module}_version)

# to get the gfalgs_build_type
default_external_project_build_type(${module})

# to get the _gflags_build_type shared or static
project_build_shared(${module})

message("gflags: version->${${module}_version} build in->${_${module}_build_type} shared?->${_${module}_build_shared}")

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
    GIT_REPOSITORY ${${module}_url}
    GIT_TAG ${${module}_tag}
    SOURCE_DIR "${external_download_dir}/${module}"
    BUILD_IN_SOURCE 0
    BUILD_BYPRODUCTS ${gflags_lib_name} ${gflags_nothreads_lib_name}
    INSTALL_COMMAND make install
    BUILD_COMMAND make -j 8
    CMAKE_CACHE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${_${module}_build_type}
        -DCMAKE_INSTALL_PREFIX:STRING=${CMAKE_SOURCE_DIR}/install
        -DBUILD_PACKAGING:BOOL=OFF
        -DBUILD_SHARED_LIBS:BOOL=${_${module}_build_shared}
        -DBUILD_TESTING:BOOL=OFF
        -DBUILD_gflags_LIB:BOOL=ON
        -DBUILD_gflags_nothreads_LIB:BOOL=ON
        -DREGISTER_BUILD_DIR:BOOL=OFF
        -DREGISTER_INSTALL_PREFIX:BOOL=OFF
)