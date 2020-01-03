# base on gtest v2.2.2
message(STATUS ${CMAKE_CURRENT_LIST_FILE})
set(module gtest)
include(popular_message)
project_base_system_message()
include(ExternalProject)

set(${module}_url https://github.com/google/googletest.git)

# this is a list contain versions which have the same behavior
include(external_setting)
set(${module}_supported_version "1.10.x" "1.8.x")
set(${module}_supported_tag "v1.10.x" "v1.8.x")
version_selector(${module} ${module}_supported_version "1.10.x")
message("${${module}_version}")
version_tag_matcher(${module} ${module}_supported_version ${module}_supported_tag ${module}_version)

# to get the gfalgs_build_type
default_external_project_build_type(${module})

project_build_shared(${module})

message("gtest: version->${${module}_version} build in->${${module}_build_type} shared?->${${module}_build_shared}")

if(${gtest_version} IN_LIST uniform_version_one)
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        if(${module}_build_type STREQUAL "Debug")
            set(gtest_lib_name gtestd.lib)
            set(gmock_lib_name gmockd.lib)
        elseif(${module}_build_type STREQUAL "Release")
            set(gtest_lib_name gtest.lib)
            set(gmock_lib_name gmock.lib)
        endif()
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        if(${module}_build_type STREQUAL "Debug")
            if(${module}_build_shared)
                set(gtest_lib_name libgtestd.so)
                set(gmock_lib_name libgmockd.so)
            else()
                set(gtest_lib_name libgtestd.a)
                set(gmock_lib_name libgmockd.a)
            endif()
        elseif(${module}_build_type STREQUAL "Release")
            if(${module}_build_shared)
                set(gtest_lib_name libgtest.so)
                set(gmock_lib_name libgmock.so)
            else()
                set(gtest_lib_name libgtest.a)
                set(gmock_lib_name libgmock.a)
            endif()
        endif()
    endif()
else()
endif()

set(${module}_include ${external_install_path}/include)
set(${module}_lib_dir ${external_install_path}/lib)
set(${module}_lib ${gtest_lib_dir}/${gtest_lib_name})

ExternalProject_Add(
    ${module}
    PREFIX ${module} 
    GIT_REPOSITORY ${${module}_url}
    GIT_TAG ${${module}_tag}
    DOWNLOAD_DIR ${external_download_dir}
    BUILD_IN_SOURCE 0
    INSTALL_COMMAND make install
    BUILD_COMMAND make -j 8
    CMAKE_CACHE_ARGS
        -DBUILD_GMOCK:BOOL=ON
        -DBUILD_SHARED_LIBS:BOOL=${_gtest_build_shared}
        -DCMAKE_BUILD_TYPE:STRING=${gtest_build_type}
        -DCMAKE_INSTALL_PREFIX:STRING=${CMAKE_SOURCE_DIR}/install
        -DINSTALL_GTEST:BOOL=ON
        -Dgmock_build_tests:BOOL=OFF
        -Dgtest_build_samples:BOOL=OFF
        -Dgtest_build_tests:BOOL=OFF
        -Dgtest_disable_pthreads:BOOL=OFF
        -Dgtest_force_shared_crt:BOOL=ON
        -Dgtest_hide_internal_symbols:BOOL=OFF
)