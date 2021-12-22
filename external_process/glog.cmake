# base on glog v2.2.2
function(glog_target)
    include(popular_message)
    include(external_setting)
    # cmakelists_base_header()
    # project_base_system_message()

    include(ExternalProject)
    include(fold_operation)

    set(module glog)
    parser_the_arguments(${module} git)


    set(${module}_url https://github.com/google/glog.git)
    set(${module}_supported_version 0.4.0)
    set(${module}_supported_tag v0.4.0)
    version_selector(${module} ${module}_supported_version 0.4.0)
    version_tag_matcher(${module} ${module}_supported_version ${module}_supported_tag ${module}_version)
    default_external_project_build_type(${module})
    project_build_shared(${module})
    cmake_external_project_common_args(${module})

    find_package(${module} ${${module}_version} CONFIG NO_CMAKE_PACKAGE_REGISTRY PATHS ${${module}_cmake_install_prefix})

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
        message(DEBUG "GLOG FOUND(TEMP)")
        fix_glog_target_name()
        set(${module}_target_name glog::glog PARENT_SCOPE)
    else()
        message(DEBUG "GLOG NOT FOUND(TEMP)")

        string(TOUPPER ${_${module}_build_type} _${module}_build_type)
        if(_${module}_build_type STREQUAL "RELEASE")
            if(_${module}_build_shared)
                set(location ${${module}_cmake_install_prefix}/lib/libglog.so)
            else()
                set(location ${${module}_cmake_install_prefix}/lib/libglog.a)
            endif()
        else()
            if(_${module}_build_shared)    
                set(location ${${module}_cmake_install_prefix}/lib/libglogd.so)
            else()
                set(location ${${module}_cmake_install_prefix}/lib/libglogd.a)
            endif()
        endif()

        ExternalProject_Add(
            _${module}
            PREFIX ${module} 
            GIT_REPOSITORY ${${module}_url}
            GIT_TAG ${${module}_tag}
            SOURCE_DIR ${${module}_source_dir}
            BUILD_IN_SOURCE ${${module}_build_in_source}
            BINARY_DIR ${${module}_binary_dir}
            INSTALL_COMMAND ${${module}_install_command}
            BUILD_COMMAND ${${module}_build_command}
            CMAKE_GENERATOR ${${module}_cmake_generator}
            CMAKE_GENERATOR_TOOLSET ${${module}_cmake_generator_toolset}
            CMAKE_GENERATOR_PLATFORM ${${module}_cmake_generator_platform}
            CMAKE_CACHE_ARGS
                -DBUILD_SHARED_LIBS:BOOL=${_${module}_build_shared}
                -DCMAKE_BUILD_TYPE:STRING=${_${module}_build_type}
                -DCMAKE_INSTALL_PREFIX:STRING=${${module}_cmake_install_prefix}
                -DCMAKE_INSTALL_PREFIX:STRING=${external_install_path}
                -DBUILD_TESTING:BOOL=OFF
                -DWITH_GFLAGS:BOOL=OFF
                -DWITH_THREADS:BOOL=ON
                -DWITH_TLS:BOOL=ON
        )

        # make sure the dir exist
        set(${module}_include_dir ${external_install_path}/include)
        set(${module}_lib_dir ${external_install_path}/lib)
        touch_fold(${module}_include_dir)
        touch_fold(${module}_lib_dir)

        add_library(glog::glog UNKNOWN IMPORTED)
        set_target_properties(
            ${${module}_target_name}
            PROPERTIES
            IMPORTED_LOCATION_${_${module}_build_type} "${location}"
        )
        add_library(interface_lib${module} INTERFACE)
        target_link_libraries(interface_lib${module} 
            INTERFACE
            glog::glog
        )
        set_target_properties(
            interface_lib${module}
            INTERFACE_INCLUDE_DIRECTORIES "${${module}_include_dir}"
            # todo: the link
            # INTERFACE_LINK_LIBRARIES "-lpthread"
            INTERFACE_LINK_DIRECTORIES "${${module}_lib_dir}"
            INTERFACE_COMPILE_DEFINITIONS "GOOGLE_GLOG_DLL_DECL;GOOGLE_GLOG_DLL_DECL_FOR_UNITTESTS"
        )
        add_dependencies(interface_lib${module} _${module})
        set(${module}_target_name interface_lib${module} PARENT_SCOPE)
    endif()
endfunction(glog_target)