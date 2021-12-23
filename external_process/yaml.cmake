################################################
# the third party yaml target construct
# generate target: ${GTest_target_name}
################################################
# base on yaml v2.2.2
#[[
    level[PRIVATE|PUBLIC]
        * PRIVATE: the target dependence would not be install
        * PUBLIC: the target dependence would be install
]]
function(yaml_target)
    include(popular_message)
    include(external_setting)
    #cmakelists_base_header()
    #project_base_system_message()

    include(ExternalProject)
    include(fold_operation)

    set(module yaml)
    parser_the_arguments(${module} git)

    # this is a list contain versions which have the same behavior
    set(${module}_url https://github.com/jbeder/yaml-cpp.git)
    set(${module}_supported_version 0.6.3)
    set(${module}_supported_tag yaml-cpp-0.6.3)
    version_selector(${module} ${module}_supported_version 0.6.3)
    version_tag_matcher(${module} ${module}_supported_version ${module}_supported_tag ${module}_version)

    default_external_project_build_type(${module})
    project_build_shared(${module})
    cmake_external_project_common_args(${module})
    message("yaml: version->${${module}_version} build in->${${${module}_build_type_var_name}} shared?->${${${module}_build_shared_var_name}}")

    # todo: 查找依赖，理应对version build_type build_shared进行适配，目前只设配了version
    find_package(yaml-cpp ${${module}_version} CONFIG NO_CMAKE_PACKAGE_REGISTRY PATHS ${${module}_cmake_install_prefix} ${${module}_binary_dir})

    if(yaml-cpp_FOUND)
        message(STATUS "Thirdparty ${module} Found")
        add_library(interface_lib${module} INTERFACE)
        target_link_libraries(interface_lib${module} INTERFACE yaml-cpp)
    else()
        # todo: 存在一个问题，如果git repo已经在本地存在，它仍然会重新clone？
        ExternalProject_Add(
            _${module}
            PREFIX ${${module}_prefix}
            GIT_REPOSITORY ${${module}_url}
            GIT_TAG ${${module}_tag}
            SOURCE_DIR ${${module}_source_dir}
            BUILD_IN_SOURCE ${${module}_build_in_source}
            BINARY_DIR ${${module}_binary_dir}
            #INSTALL_COMMAND "${${module}_install_command}"
            #BUILD_COMMAND "${${module}_build_command}"
            UPDATE_COMMAND ""
            CMAKE_GENERATOR ${${module}_cmake_generator}
            CMAKE_GENERATOR_TOOLSET ${${module}_cmake_generator_toolset}
            CMAKE_GENERATOR_PLATFORM ${${module}_cmake_generator_platform}
            CMAKE_CACHE_ARGS
                -DBUILD_GMOCK:BOOL=ON
                -DBUILD_SHARED_LIBS:BOOL=${_${module}_build_shared}
                -DCMAKE_BUILD_TYPE:STRING=${_${module}_build_type}
                -DCMAKE_INSTALL_PREFIX:STRING=${${module}_cmake_install_prefix}
                -DYAML_BUILD_SHARED_LIBS:BOOL=${_${module}_build_shared}
                -DYAML_CPP_BUILD_TESTS:BOOL=OFF
                -DYAML_MSVC_SHARED_RT:BOOL=ON
                -DYAML_MSVC_STHREADED_RT:BOOL=OFF
        )

        string(TOUPPER ${_${module}_build_type} _${module}_build_type)
        if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
            if(_${module}_build_type STREQUAL "DEBUG")
                set(yaml_lib_name libyaml-cppmdd.lib)
            elseif(_${module}_build_type STREQUAL "RELEASE")
                set(yaml_lib_name libyaml-cppmd.lib)
            endif()
        elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
            if(_${module}_build_type STREQUAL "DEBUG")
                if(_${module}_build_shared)
                    set(yaml_lib_name libyaml-cppd.so)
                else()
                    set(yaml_lib_name libyaml-cppd.lib)
                endif()
            elseif(_${module}_build_type STREQUAL "RELEASE")
                if(_${module}_build_shared)
                    set(yaml_lib_name libyaml-cpp.so)
                else()
                    set(yaml_lib_name libyaml-cpp.lib)
                endif()
            endif()
        else()
            message(FATAL_ERROR error occur while getting system: ${CMAKE_SYSTEM_NAME})
        endif()

        set(${module}_definition "")
        set(${module}_include ${${module}_cmake_install_prefix}/include)
        set(${module}_lib_dir ${${module}_cmake_install_prefix}/lib)
        set(${module}_yaml_lib ${${module}_lib_dir}/${yaml_lib_name})
        if(_${${module}_build_shared})
            set(${module}_definition "")
        else()
        endif()
        add_library(yaml::yaml UNKNOWN IMPORTED)
        set_property(TARGET yaml::yaml APPEND PROPERTY IMPORTED_CONFIGURATIONS ${_${module}_build_type})
        set_target_properties(
            yaml::yaml
            PROPERTIES
            IMPORTED_LOCATION_${_${module}_build_type} "${${module}_yaml_lib}"
        )
        add_library(interface_lib${module} INTERFACE)
        target_link_libraries(
            interface_lib${module}
            INTERFACE
            yaml::yaml
        )
        set_target_properties(
            interface_lib${module}
            PROPERTIES
            INTERFACE_COMPILE_DEFINITIONS "${${module}_definition}"
            INTERFACE_INCLUDE_DIRECTORIES "${${module}_include}"
            )
        add_dependencies(interface_lib${module} PRIVATE _${module})
    endif()
    set(yaml_module_name ${module} PARENT_SCOPE)
    set(${module}_target_name interface_lib${module} PARENT_SCOPE)
endfunction(yaml_target)