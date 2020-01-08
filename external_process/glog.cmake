# base on glog v2.2.2
include(popular_message)
include(external_setting)
cmakelists_base_header()
project_base_system_message()

set(module glog)
set(${module}_supported_version 0.4.0)
version_selector(${module} ${module}_supported_version 0.4.0)

find_package(${module} ${${module}_version} CONFIG NO_CMAKE_PACKAGE_REGISTRY PATHS ${external_install_path})


function(fix_glog_target_name)
    get_target_property(i glog::glog IMPORTED)
    message(STATUS "IMPORTED(glog::glog):${i}")
    get_target_property(ic glog::glog IMPORTED_CONFIGURATIONS)
    message(STATUS "IMPORTED_CONFIGURATIONS:${ic}")
    get_target_property(ilr glog::glog IMPORTED_LOCATION_${ic})
    message(STATUS "IMPORTED_LOCATION_${ic}:${ilr}")
    get_target_property(isr glog::glog IMPORTED_SONAME_${ic})
    message(STATUS "IMPORTED_SONAME_${ic}:${isr}")
    get_target_property(icd glog::glog INTERFACE_COMPILE_DEFINITIONS)
    message(STATUS "INTERFACE_COMPILE_DEFINITIONS:${icd}")
    get_target_property(iid glog::glog INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "INTERFACE_INCLUDE_CIRECTORIES:${iid}")
    get_target_property(ill glog::glog INTERFACE_LINK_LIBRARIES)
    message(STATUS "INTERFACE_LINK_LIBRARIES:${ill}")
endfunction(fix_glog_target_name)

if(${${module}_FOUND})
    fix_glog_target_name()
    add_custom_target(${module})
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
            -DCMAKE_INSTALL_PREFIX:STRING=${external_install_path}
            -DBUILD_TESTING:BOOL=OFF
            -DWITH_GFLAGS:BOOL=OFF
            -DWITH_THREADS:BOOL=ON
            -DWITH_TLS:BOOL=ON
    )
    #find_package(${module} ${${module}_version} CONFIG NO_CMAKE_PACKAGE_REGISTRY PATHS ${external_install_path})
    #fix_glog_target_name()
endif()
set(${module}_target_name glog)
message(STATUS "this external target's name is ${${module}_target_name}")