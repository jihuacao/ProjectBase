# base on glog v2.2.2
include(popular_message)
include(external_setting)
cmakelists_base_header()
project_base_system_message()
include(fold_operation)

set(module protobuf)
set(${module}_target_name protobuf)
set(generate_${module}_op_name _${module})
set(generate_${module}_imported_name ${module}_generate_target)
set(${module}_url https://github.com/protocolbuffers/protobuf.git)
set(${module}_supported_version 3.5.2 3.8.0)
set(${module}_supported_tag 3.5.x 3.8.x)

version_selector(${module} ${module}_supported_version 3.5.2)

default_external_project_build_type(${module})

project_build_shared(${module})

message(STATUS "finding ${module} with version: ${${module}_version}")
find_package(${module} ${${module}_version} CONFIG NO_CMAKE_PACKAGE_REGISTRY PATHS ${external_install_path})

function(fix_protobuf_target_name)
endfunction(fix_protobuf_target_name)

function(download_and_build_protobuf)
endfunction(download_and_build_protobuf)

if(${${module}_FOUND})
    message(STATUS "${module} Found for  ${${module}_version}")
    fix_protobuf_target_name()
    set(${module}_target_name protobuf::libprotobuf CACHE INTERNAL "module target name")
    set(${module}_target_name-lite protobuf::libprotobuf-lite CACHE INTERNAL "module target name")
else()
    message(STATUS "PROTOBUF NOT FOUND(TEMP)")
    include(ExternalProject)

    version_tag_matcher(${module} ${module}_supported_version ${module}_supported_tag ${module}_version)

    set(${module}_include ${external_install_path}/include)
    set(${module}_lib_dir ${external_install_path}/lib)
    if(_${module}_build_type STREQUAL "RELEASE")
        if(_${module}_build_shared)
            set(soname libprotobuf.so)
            set(soname-lite libprotobuf-lite.so)
            set(soname-c libprotoc.so)
            set(location ${external_install_path}/lib/libprotobuf.so)
            set(location-lite ${external_install_path}/lib/libprotobuf-lite.so)
            set(location-c ${external_install_path}/lib/libprotoc.so)
        else()
            set(soname libprotobufd.so)
            set(soname-lite libprotobuf-lited.so)
            set(soname-c libprotocd.so)
            set(location ${external_install_path}/lib/libprotobufd.so)
            set(location-lite ${external_install_path}/lib/libprotobuf-lited.so)
            set(location-c ${external_install_path}/lib/libprotocd.so)
        endif()
    else()
        if(_${module}_build_shared)
            set(soname libprotobuf.a)
            set(soname-lite libprotobuf-lite.a)
            set(soname-c libprotoc.a)
            set(location ${external_install_path}/lib/libprotobuf.a)
            set(location-lite ${external_install_path}/lib/libprotobuf-lite.a)
            set(location-c ${external_install_path}/lib/libprotoc.a)
        else()
            set(soname libprotobufd.a)
            set(soname-lite libprotobuf-lited.a)
            set(soname-c libprotocd.a)
            set(location ${external_install_path}/lib/libprotobufd.a)
            set(location-lite ${external_install_path}/lib/libprotobuf-lited.a)
            set(location-c ${external_install_path}/lib/libprotocd.a)
        endif()
    endif()

#第一套构建方法，不使用cmake
#    ExternalProject_Add(
#        ${generate_${module}_op_name}
#
#        PREFIX ${module}
#        SOURCE_DIR "${external_download_dir}/${module}"
#        GIT_REPOSITORY ${${module}_url}
#        GIT_TAG ${${module}_tag}
#        GIT_SHALLOW ON
#        GIT_PROCESS 8
#
#        UPDATE_DISCONNECTED ON
#
#        CONFIGURE_COMMAND ${CMAKE_COMMAND} -E chdir ${external_download_dir}/${module} ./autogen.sh
#        COMMAND ${CMAKE_COMMAND} -E chdir ${external_download_dir}/${module} ./configure --prefix=${external_install_path}
#
#        BUILD_COMMAND ${CMAKE_COMMAND} -E chdir ${external_download_dir}/${module} make -j 8
#
#        TEST_COMMAND ${CMAKE_COMMAND} -E chdir ${external_download_dir}/${module} make check
#
#        INSTALL_COMMAND ${CMAKE_COMMAND} -E chdir ${external_download_dir}/${module} make install
#        #COMMAND ${CMAKE_COMMAND} -E chdir ${external_install_path} mkdir include/google/protobuf/cmake
#        COMMAND ${CMAKE_COMMAND} -E chdir ${external_download_dir}/${module} cp -rf ./cmake ${external_install_path}/include/google/protobuf
#    )

#第二套构建方法，使用cmake
    set(build_fold_name build-${_${module}_build_type}-${_${module}_build_shared})
    set(build_dir ${external_download_dir}/${module}/cmake/build-${_${module}_build_type}-${_${module}_build_shared})
    message(STATUS "build ${module} in ${build_dir}")
    set(cmake_dir ..)
    ExternalProject_Add(
        ${generate_${module}_op_name}

        PREFIX ${module}
        SOURCE_DIR "${external_download_dir}/${module}"

        GIT_REPOSITORY ${${module}_url}
        GIT_TAG ${${module}_tag}
        GIT_SHALLOW ON
        GIT_PROCESS 8

        UPDATE_DISCONNECTED ON

        CMAKE_COMMAND ${CMAKE_COMMAND} -E chdir ${build_dir} cmake ${cmake_dir}
        CMAKE_CACHE_ARGS
            -DCMAKE_BUILD_TYPE:STRING=${_${module}_build_type}
            -DCMAKE_INSTALL_PREFIX:STRING=${external_install_path}
            -Dprotobuf_WITH_ZLIB:BOOL=ON
            -Dprotobuf_BUILD_SHARED_LIBS:BOOL=${_${module}_build_shared}
            -Dprotobuf_BUILD_EXAMPLES:BOOL=ON
            -Dprotobuf_BUILD_TESTS:BOOL=OFF
            -Dprotobuf_INSTALL_EXAMPLES:BOOl=OFF
        
        BUILD_COMMAND ${CMAKE_COMMAND} -E chdir ${build_dir} make -j 8

        #TEST_COMMAND ${CMAKE_COMMAND} -E chdir ${build_dir} make check

        INSTALL_COMMAND ${CMAKE_COMMAND} -E chdir ${build_dir} make install
    )
    ExternalProject_Add_Step(
        ${generate_${module}_op_name}

        mkbuilddir

        DEPENDEES patch
        DEPENDERS configure

        COMMAND ${CMAKE_COMMAND} -E chdir ${external_download_dir}/${module}/cmake rm -rf ${build_dir}
        COMMAND ${CMAKE_COMMAND} -E chdir ${external_download_dir}/${module}/cmake mkdir ${build_dir}
    )
    add_library(
        ${generate_${module}_imported_name} 
        INTERFACE 
        IMPORTED
    )
    set_property(TARGET ${generate_${module}_imported_name} APPEND PROPERTY IMPORTED_CONFIGURATIONS ${_${module}_build_type})
    set_target_properties(
        ${generate_${module}_imported_name}
        PROPERTIES
        INTERFACE_COMPILE_DEFINITIONS "GTEST_LINKED_AS_SHARED_LIBRARY=${link_as_shared}"
        INTERFACE_INCLUDE_DIRECTORIES "${${module}_include}"
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${${module}_include}"
        IMPORTED_LINK_INTERFACE_LIBRARIES_${_${module}_build_type} "${dep}"
        IMPORTED_LOCATION_${_${module}_build_type} "${location};${location-lite};${location-c}"
        IMPORTED_SONAME_${_${module}_build_type} "${soname};${soname-lite};${soname-c}"
    )
    add_dependencies(
        ${generate_${module}_imported_name} 
        ${generate_${module}_op_name}
    )
    ## make sure the dir exist
    #set(include_dir ${external_install_path}/include)
    #set(lib_dir ${external_install_path}/lib)
    #touch_fold(include_dir)
    #touch_fold(lib_dir)

    #add_library(${${module}_target_name} UNKNOWN IMPORTED GLOBAL)
    #set_target_properties(
    #    ${${module}_target_name}
    #    PROPERTIES
    #    INTERFACE_INCLUDE_DIRECTORIES "${include_dir}"
    #    INTERFACE_LINK_LIBRARIES "-lpthread"
    #    INTERFACE_LINK_DIRECTORIES "${lib_dir}"
    #    INTERFACE_COMPILE_DEFINITIONS "GOOGLE_GLOG_DLL_DECL=;GOOGLE_GLOG_DLL_DECL_FOR_UNITTESTS="
    #    IMPORTED_LOCATION_${_${module}_build_type} "${location}"
    #    IMPORTED_SONAME_${_${module}_build_type} "${soname}"
    #)
    #add_dependencies(${${module}_target_name} ${module})
    unset(location)
    unset(soname)
    unset(include_dir)
    unset(lib_dir)
    unset(build_fold_name)
    unset(build_dir)
    unset(cmake_dir)
    set(${module}_target_name ${generate_${module}_imported_name} CACHE INTERNAL "module target name")
    set(${module}_target_name-lite ${generate_${module}_imported_name} CACHE INTERNAL "module target name")
endif()
unset(generate_${module}_op_name)
message(STATUS "this external target's name is ${${module}_target_name}")