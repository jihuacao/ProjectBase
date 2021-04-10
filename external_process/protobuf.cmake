# base on glog v2.2.2
include(popular_message)
include(external_setting)
cmakelists_base_header()
project_base_system_message()
include(fold_operation)

set(module protobuf)

set(${module}_target_name protobuf)

set(${module}_url https://github.com/protocolbuffers/protobuf.git)

set(${module}_supported_version 3.5.x)

set(${module}_supported_tag 3.5.x)

version_selector(${module} ${module}_supported_version 3.5.x)

default_external_project_build_type(${module})

project_build_shared(${module})

message(STATUS "tt:${external_install_path}")
find_package(${module} ${${module}_version} CONFIG NO_CMAKE_PACKAGE_REGISTRY PATHS ${external_install_path})

function(fix_protobuf_target_name)
endfunction(fix_protobuf_target_name)

function(download_and_build_protobuf)
endfunction(download_and_build_protobuf)

if(${${module}_FOUND})
    message(STATUS "${module} Found for  ${${module}_version}")
    fix_protobuf_target_name()
else()
    message(STATUS "PROTOBUG NOT FOUND(TEMP)")
    include(ExternalProject)

    version_tag_matcher(${module} ${module}_supported_version ${module}_supported_tag ${module}_version)

    string(TOUPPER ${_${module}_build_type} _${module}_build_type)
    if(_${module}_build_type STREQUAL "RELEASE")
        if(_${module}_build_shared)
            set(soname liblog.so)
            set(location ${external_install_path}/lib/libglog.so)
        else()
            set(soname liblog.a)
            set(location ${external_install_path}/lib/libglog.a)
        endif()
    else()
        if(_${module}_build_shared)    
            set(soname liblogd.so)
            set(location ${external_install_path}/lib/libglogd.so)
        else()
            set(soname liblogd.a)
            set(location ${external_install_path}/lib/libglogd.a)
        endif()
    endif()

    add_custom_target(${generate_protobuf_op_name}
        COMMAND ${CMAKE_COMMAND} -E chdir ${external_download_dir} git clone https://github.com/protocolbuffers/protobuf.git
        COMMAND ${CMAKE_COMMAND} -E chdir ${external_download_dir} git submodule update --init --resurvise
        COMMAND ${CMAKE_COMMAND} -E chdir ${external_download_dir} git checkout ${module}_version
        COMMAND ${CMAKE_COMMAND} -E chdir ${external_download_dir} ./autogen.sh
        COMMAND ${CMAKE_COMMAND} -E chdir ${external_download_dir} ./configure --prefix=${external_install_path}
        COMMAND ${CMAKE_COMMAND} -E chdir ${external_download_dir}
        COMMAND ${CMAKE_COMMAND} -E chdir ${external_download_dir}
        COMMAND ${CMAKE_COMMAND} -E chdir ${external_download_dir}
    )
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

    # make sure the dir exist
    set(include_dir ${external_install_path}/include)
    set(lib_dir ${external_install_path}/lib)
    touch_fold(include_dir)
    touch_fold(lib_dir)

    add_library(${${module}_target_name} UNKNOWN IMPORTED GLOBAL)
    set_target_properties(
        ${${module}_target_name}
        PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${include_dir}"
        INTERFACE_LINK_LIBRARIES "-lpthread"
        INTERFACE_LINK_DIRECTORIES "${lib_dir}"
        INTERFACE_COMPILE_DEFINITIONS "GOOGLE_GLOG_DLL_DECL=;GOOGLE_GLOG_DLL_DECL_FOR_UNITTESTS="
        IMPORTED_LOCATION_${_${module}_build_type} "${location}"
        IMPORTED_SONAME_${_${module}_build_type} "${soname}"
    )
    add_dependencies(${${module}_target_name} ${module})
    unset(location)
    unset(soname)
    unset(include_dir)
    unset(lib_dir)

endif()
message(STATUS "this external target's name is ${${module}_target_name}")