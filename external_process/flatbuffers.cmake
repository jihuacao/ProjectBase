﻿# base on flatbuffers v2.2.2
function(flatbuffers_target)
    include(popular_message)
    include(external_setting)
    # cmakelists_base_header()
    # project_base_system_message()

    include(ExternalProject)
    include(fold_operation)

    set(module flatbuffers)
    parser_the_arguments(${module} git)

    set(${module}_url https://github.com/google/flatbuffers.git)
    set(${module}_supported_version 2.0.5)
    set(${module}_supported_tag v2.0.5)
    version_selector(${module} ${module}_supported_version 2.0.5)
    version_tag_matcher(${module} ${module}_supported_version ${module}_supported_tag ${module}_version)
    cmake_external_project_common_args(${module})
    message("${module}: version->${${module}_version} build in->${${${module}_build_type_var_name}} shared?->${${${module}_build_shared_var_name}}")

    find_package(Flatbuffers ${${module}_version} CONFIG NO_CMAKE_PACKAGE_REGISTRY PATHS ${${module}_cmake_install_prefix})

    if(${Flatbuffers_FOUND})
        message(DEBUG "${module} FOUND(TEMP)")
        add_library(interface_lib${module} INTERFACE)
        target_link_libraries(interface_lib${module} INTERFACE flatbuffers::flatbuffers)
    else()
        message(DEBUG "${module} NOT FOUND(TEMP)")
        ExternalProject_Add(
            ext_${module}
            "${${module}_external_project_add_args}"
            PREFIX ${module} 
            "${${module}_external_project_add_git_args}"
            GIT_REPOSITORY ${${module}_url}
            GIT_TAG ${${module}_tag}
            CMAKE_ARGS
                "${${module}_cmake_args}"
            CMAKE_CACHE_ARGS
                -DBUILD_SHARED_LIBS:BOOL=${${${module}_build_shared_var_name}}
                #-DCMAKE_BUILD_TYPE:STRING=${${${module}_build_type_var_name}}
                #-DCMAKE_INSTALL_PREFIX:STRING=${${module}_cmake_install_prefix}
                -DBUILD_TESTING:BOOL=OFF
                -DWITH_GFLAGS:BOOL=OFF
                -DWITH_THREADS:BOOL=ON
                -DBUILD_TESTING:BOOL=OFF
                -DWITH_TLS:BOOL=ON
        )

        # make sure the dir exist
        set(${module}_include_dir ${${module}_cmake_install_prefix}/include)
        set(${module}_lib_dir ${${module}_cmake_install_prefix}/lib)
        touch_fold(${module}_include_dir)
        touch_fold(${module}_lib_dir)
        # todo:
        list(
            APPEND
            status_registry
            SYSTEM_NAME 
            GENERATOR 
            GENERATOR_PLATFORM
            GENERATOR_TOOLSET
            GENERATOR_INSTANCE
            BUILD_SHARED
            BUILD_TYPE
        )
        list(
            APPEND
            component_registry
            PREFIX
            POSTFIX
            EXTENSION
            DEFINITIONS
        )
        generate_object_name_component(
            status_registry
            component_registry
            SYSTEM_NAME ${CMAKE_SYSTEM_NAME}
            GENERATOR ${${module}_cmake_generator}
            GENERATOR_PLATFORM ${${module}_cmake_generator_platform}
            GENERATOR_TOOLSET ${${module}_cmake_generator_toolset}
            GENERATOR_INSTANCE ${${module}_cmake_generator_instance}
            BUILD_SHARED ${${${module}_build_shared_var_name}}
            BUILD_TYPE ${${${module}_build_type_var_name}}
            SYSTEM_NAME_LIST 
            Windows 
            Linux
            Linux
            Linux
            Linux
            GENERATOR_LIST 
            "Visual Studio"
            "Unix Makefiles"
            "Unix Makefiles"
            "Unix Makefiles"
            "Unix Makefiles"
            GENERATOR_PLATFORM_LIST x64 ANY ANY ANY ANY
            GENERATOR_TOOLSET_LIST ANY ANY ANY ANY ANY
            GENERATOR_INSTANCE_LIST ANY ANY ANY ANY ANY
            BUILD_SHARED_LIST ANY ON ON OFF OFF
            BUILD_TYPE_LIST ANY RELEASE DEBUG RELEASE DEBUG
            PREFIX_LIST Empty "lib" "lib" "lib" "lib"
            POSTFIX_LIST Empty Empty "d" Empty "d"
            EXTENSION_LIST "lib" "so" "so" "a" "a"
            DEFINITIONS_LIST
            Empty
            Empty
            Empty
            Empty
            Empty
        )
        # static: 
        # shared: Empty
        set(location ${${module}_lib_dir}/${prefix}${module}${postfix}.${extension})
        message(DEBUG "${location}")

        add_library(flatbuffers::flatbuffers UNKNOWN IMPORTED)
        set_target_properties(
            flatbuffers::flatbuffers
            PROPERTIES
            # IMPORTED_LOCATION_${${${module}_build_type_var_name}} "${location}"
            IMPORTED_LOCATION "${location}"
        )
        add_library(interface_lib${module} INTERFACE)
        target_link_libraries(interface_lib${module} 
            INTERFACE
            flatbuffers::flatbuffers
        )
        set_target_properties(
            interface_lib${module}
            PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${${module}_include_dir}"
            INTERFACE_LINK_DIRECTORIES "${${module}_lib_dir}"
            INTERFACE_COMPILE_DEFINITIONS "${definitions}"
        )
        add_dependencies(interface_lib${module} ext_${module})
    endif()
    set(${module}_module_name ${module} PARENT_SCOPE)
    set(${module}_target_name interface_lib${module} PARENT_SCOPE)
endfunction(flatbuffers_target)
