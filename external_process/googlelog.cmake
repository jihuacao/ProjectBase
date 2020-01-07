# base on glog v2.2.2
include(popular_message)
include(external_setting)
cmakelists_base_header()
project_base_system_message()

set(module glog)
set(${module}_supported_version 0.4.0)
version_selector(${module} ${module}_supported_version 0.4.0)

message(STATUS "CMAKE_PREFIX_PATH:${CMAKE_PREFIX_PATH}")
message(STATUS "CMAKE_FRAMEWORK_PATH:${CMAKE_FRAMEWORK_PATH}")
message(STATUS "CMAKE_APPBUNDLE_PATH:${CMAKE_APBUNDLE_PATH}")
find_package(${module} ${${module}_version} CONFIG NO_SYSTEM_ENVIRONMENT_PATH)# PATHS /home/sins/Download/glog/build/install)

if(${${module}_FOUND})
    message(STATUS ${${module}_FOUND})
else()
    include(ExternalProject)

    set(${module}_url https://github.com/google/glog.git)
    set(${module}_supported_tag v0.4.0)

    version_tag_matcher(${module} ${module}_supported_version ${module}_supported_tag ${module}_version)

    default_external_project_build_type(${module})
    project_build_shared(${module})

    ExternalProject_Add(
        ${module}
        PREFIX ${module}
        GIT_REPOSITORY ${${module}_url}
        GIT_TAG ${${module}_tag}
        SOURCE_DIR "${external_download_dir}/${module}"
        BUILD_IN_SOURCE 0
        BUILD_COMMAND make -j 8
        INSTALL_COMMAND make install
        CMAKE_CACHE_ARGS
            -DCMAKE_BUILD_TYPE:STRING=${_${module}_build_type}
            -DBUILD_SHARED_LIBS:BOOL=${_${module}_build_shared}
            -DCMAKE_INSTALL_PREFIX:STRING=${externale_install_path}
            -DBUILD_TESTING:BOOL=OFF
            -DWITH_GFLAGS:BOOL=OFF
            -DWITH_THREADS:BOOL=ON
            -DWITH_TLS:BOOL=ON
    )
endif()