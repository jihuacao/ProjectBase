# base on gtest v2.2.2
include(popular_message)
include(external_setting)
cmakelists_base_header()
project_base_system_message()

include(ExternalProject)
include(fold_operation)
set(module GTest)
set(gtest_module_name ${module} CACHE INTERNAL "module name")

# this is a list contain versions which have the same behavior
set(${module}_supported_version 1.10.0 1.8.0)
set(${module}_supported_tag v1.10.x v1.8.x)
version_selector(${module} ${module}_supported_version 1.10.0)

find_package(GTest ${${module}_version} CONFIG NO_CMAKE_PACKAGE_REGISTRY PATHS ${external_install_path})

if(${${module}_FOUND})
    set(${module}_target_name GTest::gmock)
else()
    set(${module}_url https://github.com/google/googletest.git)
    version_tag_matcher(${module} ${module}_supported_version ${module}_supported_tag ${module}_version)

    # to get the gfalgs_build_type
    default_external_project_build_type(${module})

    project_build_shared(${module})

    set(${module}_disable_pthreads OFF CACHE BOOL "disable thread of gtest or not")
    set_property(CACHE ${module}_disable_pthreads PROPERTY ADVANCED)
    set_property(CACHE ${module}_disable_pthreads PROPERTY STRINGS "ON" "OFF")

    message("gtest: version->${${module}_version} build in->${${module}_build_type} shared?->${${module}_build_shared}")

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

    set(${module}_include ${external_install_path}/include)
    set(${module}_lib_dir ${external_install_path}/lib)
    set(${module}_gtest_lib ${${module}_lib_dir}/${gtest_lib_name})
    set(${module}_gmock_lib ${${module}_lib_dir}/${gmock_lib_name})
    if(_${${module}_build_shared})
        set(link_as_shared, 1)
    else()
        set(link_as_shared, 0)
    endif()
    if(${${module}_disable_pthreads})
        set(dep )
    else()
        set(dep Threads::Threads)
    endif()

    ExternalProject_Add(
        _${module}
        PREFIX ${module} 
        GIT_REPOSITORY ${${module}_url}
        GIT_TAG ${${module}_tag}
        SOURCE_DIR "${external_download_dir}/${module}"
        BUILD_IN_SOURCE 0
        INSTALL_COMMAND make install
        BUILD_COMMAND make -j 8
        CMAKE_CACHE_ARGS
            -DBUILD_GMOCK:BOOL=ON
            -DBUILD_SHARED_LIBS:BOOL=${_${module}_build_shared}
            -DCMAKE_BUILD_TYPE:STRING=${_${module}_build_type}
            -DCMAKE_INSTALL_PREFIX:STRING=${CMAKE_SOURCE_DIR}/install
            -DINSTALL_GTEST:BOOL=ON
            -Dgmock_build_tests:BOOL=OFF
            -Dgtest_build_samples:BOOL=OFF
            -Dgtest_build_tests:BOOL=OFF
            -Dgtest_disable_pthreads:BOOL=${${module}_disable_pthreads}
            -Dgtest_force_shared_crt:BOOL=ON
            -Dgtest_hide_internal_symbols:BOOL=OFF
    )

    # make sure the dir existed
    touch_fold(${module}_include)
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
        INTERFACE_COMPILE_DEFINITIONS "GTEST_LINKED_AS_SHARED_LIBRARY=${link_as_shared}"
        INTERFACE_INCLUDE_DIRECTORIES "${${module}_include}"
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${${module}_include}"
        IMPORTED_LINK_INTERFACE_LIBRARIES_${_${module}_build_type} "${dep}"
        IMPORTED_LOCATION_${_${module}_build_type} "${external_install_path}/lib/${gtest_lib_name}"
        IMPORTED_SONAME_${_${module}_build_type} "${gtest_lib_name}"
        #INTERFACE_LINK_LIBRARIES "${dep}"
        #LOCATION "${external_install_path}/lib/${gtest_lib_name}"
    )
    add_library(GTest::gmock UNKNOWN IMPORTED)
    set_property(TARGET GTest::gmock APPEND PROPERTY IMPORTED_CONFIGURATIONS ${_${module}_build_type})
    set_target_properties(
        GTest::gmock
        PROPERTIES
        INTERFACE_COMPILE_DEFINITIONS "GTEST_LINKED_AS_SHARED_LIBRARY=${link_as_shared}"
        INTERFACE_INCLUDE_DIRECTORIES "${${module}_include}"
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${${module}_include}"
        IMPORTED_LINK_INTERFACE_LIBRARIES_${_${module}_build_type} "${dep};GTest::gtest"
        IMPORTED_LOCATION_${_${module}_build_type} "${external_install_path}/lib/${gmock_lib_name}"
        IMPORTED_SONAME_${_${module}_build_type} "${gmock_lib_name}"
    )
    add_dependencies(GTest::gmock _${module})
    unset(link_as_shared)
    set(${module}_target_name GTest::gmock CACHE INTERNAL "module target name")
endif()