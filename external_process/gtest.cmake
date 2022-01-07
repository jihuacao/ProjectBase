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
    #cmakelists_base_header()
    #project_base_system_message()

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
    default_external_project_build_type(${module})
    project_build_shared(${module})
    cmake_external_project_common_args(${module})

    find_package(GTest ${${module}_version} CONFIG NO_CMAKE_PACKAGE_REGISTRY PATHS ${${module}_cmake_install_prefix} ${${module}_binary_dir})

    if(GTest_FOUND)
        message(STATUS "Thirdparty ${module} Found")
        add_library(interface_lib${module} INTERFACE)
        target_link_libraries(interface_lib${module} INTERFACE GTest::gmock)
    else()
        set(${module}_disable_pthreads OFF CACHE BOOL "disable thread of gtest or not")
        set_property(CACHE ${module}_disable_pthreads PROPERTY ADVANCED)
        set_property(CACHE ${module}_disable_pthreads PROPERTY STRINGS "ON" "OFF")

        message("gtest: version->${${module}_version} build in->${${module}_build_type} shared?->${${module}_build_shared}")

        # todo: 存在一个问题，如果git repo已经在本地存在，它仍然会重新clone？
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
                -DBUILD_GMOCK:BOOL=ON
                -DBUILD_SHARED_LIBS:BOOL=${_${module}_build_shared}
                -DCMAKE_BUILD_TYPE:STRING=${_${module}_build_type}
                -DCMAKE_INSTALL_PREFIX:STRING=${${module}_cmake_install_prefix}
                -DINSTALL_GTEST:BOOL=ON
                -Dgmock_build_tests:BOOL=OFF
                -Dgtest_build_samples:BOOL=OFF
                -Dgtest_build_tests:BOOL=OFF
                -Dgtest_disable_pthreads:BOOL=${${module}_disable_pthreads}
                -Dgtest_force_shared_crt:BOOL=ON
                -Dgtest_hide_internal_symbols:BOOL=OFF
        )

        string(TOUPPER ${_${module}_build_type} _${module}_build_type)
        if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
            if(_${module}_build_type STREQUAL "DEBUG")
                set(gtest_lib_name gtestd.lib)
                set(gmock_lib_name gmockd.lib)
            elseif(_${module}_build_type STREQUAL "RELEASE")
                set(gtest_lib_name gtest.lib)
                set(gmock_lib_name gmock.lib)
            endif()
        elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
            if(_${module}_build_type STREQUAL "DEBUG")
                if(_${module}_build_shared)
                    set(gtest_lib_name libgtestd.so)
                    set(gmock_lib_name libgmockd.so)
                else()
                    set(gtest_lib_name libgtestd.a)
                    set(gmock_lib_name libgmockd.a)
                endif()
            elseif(_${module}_build_type STREQUAL "RELEASE")
                if(_${module}_build_shared)
                    set(gtest_lib_name libgtest.so)
                    set(gmock_lib_name libgmock.so)
                else()
                    set(gtest_lib_name libgtest.a)
                    set(gmock_lib_name libgmock.a)
                endif()
            endif()
        else()
            message(FATAL_ERROR error occur while getting system: ${CMAKE_SYSTEM_NAME})
        endif()

        set(${module}_definition "")
        set(${module}_include_dir ${${module}_cmake_install_prefix}/include)
        set(${module}_lib_dir ${${module}_cmake_install_prefix}/lib)
        set(${module}_gtest_lib ${${module}_lib_dir}/${gtest_lib_name})
        set(${module}_gmock_lib ${${module}_lib_dir}/${gmock_lib_name})
        if(_${${module}_build_shared})
            set(${module}_definition "${${module}_definition};GTEST_LINKED_AS_SHARED_LIBRARY")
        else()
        endif()
        if(${${module}_disable_pthreads})
            set(dep )
        else()
            set(dep Threads::Threads)
        endif()
        # make sure the dir existed
        touch_fold(${module}_include_dir)
        touch_fold(${module}_lib_dir)
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
        add_dependencies(interface_lib${module} PRIVATE _${module})
    endif()
    set(gtest_module_name ${module} PARENT_SCOPE)
    set(${module}_target_name interface_lib${module} PARENT_SCOPE)
    #set(gtest_module_name ${module} CACHE INTERNAL "module name")
    #set(${module}_target_name interface_lib${module} CACHE INTERNAL "")
endfunction(gtest_target)