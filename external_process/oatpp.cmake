# base on oatpp v2.2.2
function(oatpp_target)
    include(popular_message)
    include(external_setting)
    # cmakelists_base_header()
    # project_base_system_message()

    include(ExternalProject)
    include(fold_operation)

    set(module oatpp)
    parser_the_arguments(${module} git)

    set(${module}_url https://github.com/oatpp/oatpp.git)
    set(${module}_supported_version 1.2.5)
    set(${module}_supported_tag 1.2.5)
    version_selector(${module} ${module}_supported_version 1.2.5)
    version_tag_matcher(${module} ${module}_supported_version ${module}_supported_tag ${module}_version)
    cmake_external_project_common_args(${module})
    message("${module}: version->${${module}_version} build in->${${${module}_build_type_var_name}} shared?->${${${module}_build_shared_var_name}}")

    find_package(${module} ${${module}_version} CONFIG NO_CMAKE_PACKAGE_REGISTRY PATHS ${${module}_cmake_install_prefix})

    function(fix_oatpp_target_name)
        get_target_property(i oatpp::oatpp IMPORTED)
        message(STATUS "IMPORTED(oatpp::oatpp):${i}")
        get_target_property(ic oatpp::oatpp IMPORTED_CONFIGURATIONS)
        message(STATUS "IMPORTED_CONFIGURATIONS:${ic}")
        get_target_property(ilr oatpp::oatpp IMPORTED_LOCATION_${ic})
        message(STATUS "IMPORTED_LOCATION_${ic}:${ilr}")
        get_target_property(isr oatpp::oatpp IMPORTED_SONAME_${ic})
        message(STATUS "IMPORTED_SONAME_${ic}:${isr}")
        get_target_property(icd oatpp::oatpp INTERFACE_COMPILE_DEFINITIONS)
        message(STATUS "INTERFACE_COMPILE_DEFINITIONS:${icd}")
        get_target_property(iid oatpp::oatpp INTERFACE_INCLUDE_DIRECTORIES)
        message(STATUS "INTERFACE_INCLUDE_CIRECTORIES:${iid}")
        get_target_property(ill oatpp::oatpp INTERFACE_LINK_LIBRARIES)
        message(STATUS "INTERFACE_LINK_LIBRARIES:${ill}")
    endfunction(fix_oatpp_target_name)

    if(${${module}_FOUND})
        message(DEBUG "OATPP FOUND(TEMP)")
        fix_oatpp_target_name()
        add_library(interface_lib${module} INTERFACE)
        target_link_libraries(interface_lib${module} INTERFACE oatpp::oatpp oatpp::oatpp-test)
    else()
        message(DEBUG "OATPP NOT FOUND(TEMP)")
        message(WARNING "Now oatpp only support static lib")

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
                -DOATPP_BUILD_TESTS:BOOL=OFF
        )

        # make sure the dir exist
        set(${module}_include_dir ${${module}_cmake_install_prefix}/include/oatpp-${${module}_version}/oatpp)
        set(${module}_lib_dir ${${module}_cmake_install_prefix}/lib/oatpp-${${module}_version})
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
            #[[
              * oatpp依赖了系统的sock通信，目前所知：
                * windows
                  * wsock32
                  * ws2_32
                * linux
            ]]
            DEPENDED_LIBRARIES
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
            DEPENDED_LIBRARIES_LIST
            "wsock32<list_regex>ws2_32"
            Empty
            Empty
            Empty
            Empty
        )
        # static: 
        # shared: Empty
        set(location ${${module}_lib_dir}/${prefix}${module}${postfix}.${extension})
        set(location_two ${${module}_lib_dir}/${prefix}${module}-test${postfix}.${extension})
        message(DEBUG "${location}")
        message(DEBUG "${location_two}")

        add_library(oatpp::oatpp UNKNOWN IMPORTED)
        set_target_properties(
            oatpp::oatpp
            PROPERTIES
            # IMPORTED_LOCATION_${${${module}_build_type_var_name}} "${location}"
            IMPORTED_LOCATION ${location}
        )
        target_link_libraries(
            oatpp::oatpp
            INTERFACE
            "${depended_libraries}"
        )
        add_library(oatpp::oatpp-test UNKNOWN IMPORTED)
        set_target_properties(
            oatpp::oatpp-test
            PROPERTIES
            # IMPORTED_LOCATION_${${${module}_build_type_var_name}} "${location}"
            IMPORTED_LOCATION "${location_two}"
        )
        add_library(interface_lib${module} INTERFACE)
        target_link_libraries(interface_lib${module} 
            INTERFACE
            oatpp::oatpp
            oatpp::oatpp-test
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
endfunction(oatpp_target)
