# base on gtest v2.2.2
message(${CMAKE_CURRENT_LIST_FILE})
include(popular_message)
include(ExternalProject)
project_base_system_message()

set(gtest_url https://github.com/google/googletest.git)

# this is a list contain versions which have the same behavior
set(all_version "1.10.x" "1.8.x")
set(uniform_version_one "1.10.x" "1.8.x")
set(gtest_version "1.10.x" CACHE STRING "gtest version")
set_property(CACHE gtest_version PROPERTY STRINGS ${all_version})
if(${gtest_version} STREQUAL "1.8.x")
    set(gtest_tag v1.8.x)
if(${gtest_version} STREQUAL "1.10.x")
    set(gtest_tag v1.10.x)
else()
    message(FATAL_ERROR "unsupported gtest version: ${gtest_version}")
endif()

# to get the gfalgs_build_type
set(supported_gtest_build_type "FOLLOW_CMAKE_BUILD_TYPE" "Release" "Debug")
set(gtest_build_type "FOLLOW_CMAKE_BUILD_TYPE" CACHE STRING "the specifical option for gtest, if the gtest_build_type is set")
set_property(CACHE gtest_build_type PROPERTY STRINGS "Release" "Debug" "FOLLOW_CMAKE_BUILD_TYPE")
if(gtest_build_type STREQUAL "")
    set(gtest_build_type ${CMAKE_BUILD_TYPE})
elseif(gtest_build_type STREQUAL "FOLLOW_CMAKE_BUILD_TYPE")
    set(gtest_build_type ${CMAKE_BUILD_TYPE})
elseif(${gtest_build_type} IN_LIST supported_gtest_build_type)
else()
    message(FATAL_ERROR "unsupported gtest_build_type: ${gtest_build_type}")
endif()

# to get the _gtest_build_type shared or static
set(gtest_build_shared "FOLLOW_EXTERNAL_BUILD_SHARED" CACHE STRING "specifical build the gtest in shared or follow in external_build_shared")
set_property(CACHE gtest_build_shared PROPERTY STRINGS "ON" "OFF" "FOLLOW_EXTERNAL_BUILD_SHARED")
if(gtest_build_shared STREQUAL "FOLLOW_EXTERNAL_BUILD_SHARED")
    set(_gtest_build_shared ${external_build_shared})
elseif(gtest_build_shared STREQUAL "")
    set(_gtest_build_shared ${external_build_shared})
elseif(gtest_build_shared STREQUAL "ON")
    set(_gtest_build_shared ON)
elseif(gtest_build_shared STREQUAL "OFF")
    set(_gtest_build_shared OFF)
else()
    message(FATAL_ERROR "unsupported gtest_build_shared: ${gtest_build_shared}")
endif()

message("gtest: version->4{gtest_version} build in->${gtest_build_type} shared?->${_gtest_build_shared}")

if(${gtest_version} IN_LIST uniform_version_one)
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        if(gtest_build_type STREQUAL "Debug")
            set(gtest_lib_name gtestd.lib)
            set(gmock_lib_name gmockd.lib)
        elseif(gtest_build_type STREQUAL "Release")
            set(gtest_lib_name gtest.lib)
            set(gmock_lib_name gmock.lib)
        endif()
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        if(gtest_build_type STREQUAL "Debug")
            if(_gtest_build_shared)
                set(gtest_lib_name libgtestd.so)
                set(gmock_lib_name libgmockd.so)
            else()
                set(gtest_lib_name libgtestd.a)
                set(gmock_lib_name libgmockd.a)
            endif()
        elseif(gtest_build_type STREQUAL "Release")
            if(_gtest_build_shared)
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

set(gtest_include ${external_install_path}/include)
set(gtest_lib_dir ${external_install_path}/lib)
set(gtest_lib ${gtest_lib_dir}/${gtest_lib_name})

ExternalProject_Add(gtest
    PREFIX gtest
    GIT_REPOSITORY ${gtest_url}
    GIT_TAG ${gtest_tag}
    DOWNLOAD_DIR ${external_download_dir}
    BUILD_IN_SOURCE 0
    BUILD_BYPRODUCTS ${gtest_lib_name} ${gtest_nothreads_lib_name}
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
