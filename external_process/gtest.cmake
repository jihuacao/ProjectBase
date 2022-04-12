################################################
# the third party gtest target construct
# generate target: ${GTest_target_name}
################################################
# base on gtest v2.2.2
#[[
    level[PRIVATE|PUBLIC]
        * PRIVATE: the target dependence would not be install
        * PUBLIC: the target dependence would be install
]]
function(gtest_target)
    include(popular_message)
    include(external_setting)
    include(ExternalProject)
    include(fold_operation)

    set(module gtest)
    parser_the_arguments(${module} git)

    # this is a list contain versions which have the same behavior
    set(${module}_url https://github.com/google/googletest.git)
    set(${module}_supported_version 1.10.0 1.8.0)
    set(${module}_supported_tag v1.10.x v1.8.x)
    version_selector(${module} ${module}_supported_version 1.10.0)
    version_tag_matcher(${module} ${module}_supported_version ${module}_supported_tag ${module}_version)
    cmake_external_project_common_args(${module})

    message("gtest: version->${${module}_version} build in->${${module}_build_type} shared?->${${module}_build_shared}")

    find_package(GTest ${${module}_version} CONFIG NO_CMAKE_PACKAGE_REGISTRY PATHS ${${module}_cmake_install_prefix} ${${module}_binary_dir})

    if(GTest_FOUND)
        message(STATUS "Thirdparty ${module} Found")
        add_library(interface_lib${module} INTERFACE)
        target_link_libraries(interface_lib${module} INTERFACE GTest::gmock)
    else()
        set(${module}_disable_pthreads OFF CACHE BOOL "disable thread of gtest or not")
        # [[直接确认使用pthreads]]
        set_property(CACHE ${module}_disable_pthreads PROPERTY ADVANCED)
        set_property(CACHE ${module}_disable_pthreads PROPERTY STRINGS "ON" "OFF")

        # todo: 存在一个问题，如果git repo已经在本地存在，它仍然会重新clone？
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
                -DBUILD_GMOCK:BOOL=ON
                -DBUILD_SHARED_LIBS:BOOL=${${${module}_build_shared_var_name}}
                -DINSTALL_GTEST:BOOL=ON
                -Dgmock_build_tests:BOOL=OFF
                -Dgtest_build_samples:BOOL=OFF
                -Dgtest_build_tests:BOOL=OFF
                -Dgtest_disable_pthreads:BOOL=${${module}_disable_pthreads}
                -Dgtest_force_shared_crt:BOOL=ON
                -Dgtest_hide_internal_symbols:BOOL=OFF
        )

        set(${module}_include_dir ${${module}_cmake_install_prefix}/include)
        set(${module}_lib_dir ${${module}_cmake_install_prefix}/lib)
        touch_fold(${module}_include_dir)
        touch_fold(${module}_lib_dir)

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
            LIBS
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
            Windows 
            Windows 
            Windows
            Linux
            Linux
            Linux
            Linux
            CYGWIN
            CYGWIN
            CYGWIN
            CYGWIN
            GENERATOR_LIST 
            "Visual Studio" 
            "Visual Studio" 
            "Visual Studio"
            "Visual Studio"
            "Unix Makefiles"
            "Unix Makefiles"
            "Unix Makefiles"
            "Unix Makefiles"
            "Unix Makefiles"
            "Unix Makefiles"
            "Unix Makefiles"
            "Unix Makefiles"
            GENERATOR_PLATFORM_LIST x64 x64 x64 x64 ANY ANY ANY ANY ANY ANY ANY ANY
            GENERATOR_TOOLSET_LIST ANY ANY ANY ANY ANY ANY ANY ANY ANY ANY ANY ANY
            GENERATOR_INSTANCE_LIST ANY ANY ANY ANY ANY ANY ANY ANY ANY ANY ANY ANY
            BUILD_SHARED_LIST ON ON OFF OFF ON ON OFF OFF ON ON OFF OFF
            BUILD_TYPE_LIST RELEASE DEBUG RELEASE DEBUG RELEASE DEBUG RELEASE DEBUG RELEASE DEBUG RELEASE DEBUG
            PREFIX_LIST Empty Empty Empty Empty "lib" "lib" "lib" "lib" "lib" "lib" "lib" "lib"
            POSTFIX_LIST Empty "d" Empty "d" Empty "d" Empty "d" Empty "d" Empty "d"
            EXTENSION_LIST "lib" "lib" "lib" "lib" "so" "so" "a" "a" "dll.a" "dll.a" "a" "a"
            LIBS_LIST Empty Empty Empty Empty Empty Empty Empty Empty Empty Empty Empty Empty
            DEFINITIONS_LIST
            "GTEST_LINKED_AS_SHARED_LIBRARY=1"
            "GTEST_LINKED_AS_SHARED_LIBRARY=1"
            Empty
            Empty
            "GTEST_LINKED_AS_SHARED_LIBRARY=1"
            "GTEST_LINKED_AS_SHARED_LIBRARY=1"
            Empty
            Empty
            "GTEST_LINKED_AS_SHARED_LIBRARY=1"
            "GTEST_LINKED_AS_SHARED_LIBRARY=1"
            Empty
            Empty
        )

        set(${module}_definition "")
        set(${module}_gtest_lib "${${module}_lib_dir}/${prefix}gtest${postfix}.${extension}")
        set(${module}_gmock_lib "${${module}_lib_dir}/${prefix}gmock${postfix}.${extension}")
        if(${${module}_disable_pthreads})
            set(dep )
        else()
            set(dep Threads::Threads)
        endif()
        # make sure the dir existed
        if(${${module}_disable_pthreads})
        else()
            include(CMakeFindDependencyMacro)
            find_dependency(Threads)
        endif()
        add_library(GTest::gtest UNKNOWN IMPORTED)
        set_property(TARGET GTest::gtest APPEND PROPERTY IMPORTED_CONFIGURATIONS ${_${module}_build_type})
        set_target_properties(
            GTest::gtest
            PROPERTIES
            IMPORTED_LOCATION_${_${module}_build_type} "${${module}_gtest_lib}"
        )
        add_library(GTest::gmock UNKNOWN IMPORTED)
        set_property(TARGET GTest::gmock APPEND PROPERTY IMPORTED_CONFIGURATIONS ${_${module}_build_type})
        set_target_properties(
            GTest::gmock
            PROPERTIES
            IMPORTED_LOCATION_${_${module}_build_type} "${${module}_gmock_lib}"
        )
        add_library(interface_lib${module} INTERFACE)
        target_link_libraries(
            interface_lib${module} 
            INTERFACE
            GTest::gmock
            GTest::gtest
            ${dep}
        )
        set_target_properties(
            interface_lib${module}
            PROPERTIES
            INTERFACE_COMPILE_DEFINITIONS "${${module}_definition}"
            INTERFACE_INCLUDE_DIRECTORIES "${${module}_include_dir}"
            INTERFACE_LINK_DIRECTORIES "${${module}_lib_dir}"
            )
        add_dependencies(interface_lib${module} PRIVATE ext_${module})
    endif()
    set(gtest_module_name ${module} PARENT_SCOPE)
    set(${module}_target_name interface_lib${module} PARENT_SCOPE)
endfunction(gtest_target)