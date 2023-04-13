function(protobuf_target)
    include(popular_message)
    include(external_setting)
    include(ExternalProject)
    include(fold_operation)

    set(module protobuf)
    parser_the_arguments(${module} git)

    set(${module}_url https://github.com/protocolbuffers/protobuf.git)
    set(${module}_supported_version 3.5.2 3.8.0)
    set(${module}_supported_tag 3.5.x 3.8.x)
    version_selector(${module} ${module}_supported_version 3.5.2)
    version_tag_matcher(${module} ${module}_supported_version ${module}_supported_tag ${module}_version)
    cmake_external_project_common_args(${module})
    message(STATUS "${module}: version->${${module}_version} build in->${${${module}_build_type_var_name}} shared?->${${${module}_build_shared_var_name}}")

    find_package(${module} ${${module}_version} CONFIG NO_CMAKE_PACKAGE_REGISTRY PATHS ${${module}_cmake_install_prefix})

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

        ExternalProject_Add(
            ext_${module}
            "${${module}_external_project_add_args}"
            PREFIX ${module} 
            "${${module}_external_project_add_git_args}"
            GIT_REPOSITORY ${${module}_url}
            GIT_TAG ${${module}_tag}
            CMAKE_ARGS
                "${${module}_cmake_args}"
                -S ${${module}_source_dir}/cmake/
            GIT_SHALLOW ON
            GIT_PROCESS 8
            UPDATE_DISCONNECTED ON
            CMAKE_CACHE_ARGS
                -Dprotobuf_WITH_ZLIB:BOOL=ON
                -Dprotobuf_BUILD_EXAMPLES:BOOL=ON
                -Dprotobuf_BUILD_TESTS:BOOL=OFF
                -Dprotobuf_INSTALL_EXAMPLES:BOOl=OFF
        )
        set(${module}_include_dir ${${module}_cmake_install_prefix}/include)
        set(${module}_lib_dir ${${module}_cmake_install_prefix}/lib)
        touch_fold(${module}_include_dir)
        touch_fold(${module}_lib_dir)
        list(APPEND status_registry SYSTEM_NAME GENERATOR GENERATOR_PLATFORM GENERATOR_TOOLSET GENERATOR_INSTANCE BUILD_SHARED BUILD_TYPE)
        list(APPEND component_registry PREFIX POSTFIX EXTENSION DEFINITIONS)
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
            Windows 
            Windows 
            Windows
            Linux
            Linux
            Linux
            Linux
            GENERATOR_LIST 
            "Visual Studio" 
            "Visual Studio" 
            "Visual Studio"
            "Visual Studio"
            "Unix Makefiles"
            "Unix Makefiles"
            "Unix Makefiles"
            "Unix Makefiles"
            GENERATOR_PLATFORM_LIST x64 x64 x64 x64 ANY ANY ANY ANY
            GENERATOR_TOOLSET_LIST ANY ANY ANY ANY ANY ANY ANY ANY
            GENERATOR_INSTANCE_LIST ANY ANY ANY ANY ANY ANY ANY ANY
            BUILD_SHARED_LIST ON ON OFF OFF ON ON OFF OFF
            BUILD_TYPE_LIST RELEASE DEBUG RELEASE DEBUG RELEASE DEBUG RELEASE DEBUG
            PREFIX_LIST Empty Empty Empty Empty "lib" "lib" "lib" "lib"
            POSTFIX_LIST Empty "d" Empty "d" Empty "d" Empty "d"
            EXTENSION_LIST "lib" "lib" "lib" "lib" "so" "so" "a" "a"
            DEFINITIONS_LIST
            "GTEST_LINKED_AS_SHARED_LIBRARY=${${${module}_build_shared_var_name}}"
            "GTEST_LINKED_AS_SHARED_LIBRARY=${${${module}_build_shared_var_name}}"
            "GTEST_LINKED_AS_SHARED_LIBRARY=${${${module}_build_shared_var_name}}"
            "GTEST_LINKED_AS_SHARED_LIBRARY=${${${module}_build_shared_var_name}}"
            "GTEST_LINKED_AS_SHARED_LIBRARY=${${${module}_build_shared_var_name}}"
            "GTEST_LINKED_AS_SHARED_LIBRARY=${${${module}_build_shared_var_name}}"
            "GTEST_LINKED_AS_SHARED_LIBRARY=${${${module}_build_shared_var_name}}"
            "GTEST_LINKED_AS_SHARED_LIBRARY=${${${module}_build_shared_var_name}}"
        )
        add_library(${module}_lib UNKNOWN IMPORTED)
        set_target_properties(${module}_lib PROPERTIES IMPORTED_LOCATION ${${module}_lib_dir}/${prefix}${module}${postfix}.${extension})
        add_library(${module}_lib_lite UNKNOWN IMPORTED)
        set_target_properties(${module}_lib_lite PROPERTIES IMPORTED_LOCATION ${${module}_lib_dir}/${prefix}${module}-lite${postfix}.${extension})
        add_library(${module}_lib_c UNKNOWN IMPORTED)
        set_target_properties(${module}_lib_c PROPERTIES IMPORTED_LOCATION ${${module}_lib_dir}/${prefix}${module}c${postfix}.${extension})
        add_library(${module}_interface INTERFACE)
        target_link_libraries(${module}_interface INTERFACE ${module}_lib ${module}_lib_lite ${module}_lib_c)
        set_target_properties(
            ${module}_interface
            PROPERTIES
            INTERFACE_COMPILE_DEFINITIONS "${definitions}"
            INTERFACE_INCLUDE_DIRECTORIES "${${module}_include_dir}"
            INTERFACE_LINK_DIRECTORIES "${${module}_lib_dir}"
            #IMPORTED_LINK_INTERFACE_LIBRARIES_${_${module}_build_type} "${dep}"
            #IMPORTED_LOCATION_${_${module}_build_type} "${location};${location-lite};${location-c}"
            #IMPORTED_SONAME_${_${module}_build_type} "${soname};${soname-lite};${soname-c}"
        )
        add_dependencies(${module}_interface ext_${module})
        set(${module}_target_name ${module}_interface PARENT_SCOPE)
        set(${module}_target_name-lite ${module}_interface PARENT_SCOPE)
    endif()
    set(${module}_module_name ${module} PARENT_SCOPE)
    message(STATUS "this external target's name is ${${module}_target_name}")
endfunction(protobuf_target)