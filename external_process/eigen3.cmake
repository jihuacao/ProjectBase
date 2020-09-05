# base on eigen v2.2.2
include(popular_message)
include(external_setting)
cmakelists_base_header()
project_base_system_message()
include(fold_operation)

set(module eigen3)
set(${module}_target_name Eigen3::Eigen)
set(${module}_supported_version 3.3.7)
version_selector(${module} ${module}_supported_version 3.3.7)

find_package(Eigen3 ${${module}_version} CONFIG NO_CMAKE_PACKAGE_REGISTRY PATHS ${external_install_path})

function(fix_eigen_target_name)
    get_target_property(i Eigen3::Eigen IMPORTED)
    message(STATUS "IMPORTED(Eigen3::Eigen):${i}")
    get_target_property(ic Eigen3::Eigen IMPORTED_CONFIGURATIONS)
    message(STATUS "IMPORTED_CONFIGURATIONS:${ic}")
    get_target_property(icd Eigen3::Eigen INTERFACE_COMPILE_DEFINITIONS)
    message(STATUS "INTERFACE_COMPILE_DEFINITIONS:${icd}")
    get_target_property(iid Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "INTERFACE_INCLUDE_CIRECTORIES:${iid}")
    get_target_property(ill Eigen3::Eigen INTERFACE_LINK_LIBRARIES)
    message(STATUS "INTERFACE_LINK_LIBRARIES:${ill}")
endfunction(fix_eigen_target_name)

if(${Eigen3_FOUND})
    fix_eigen_target_name()
else()
    include(ExternalProject)

    set(${module}_url https://gitlab.com/libeigen/eigen.git)
    set(${module}_supported_tag 3.3.7)

    version_tag_matcher(${module} ${module}_supported_version ${module}_supported_tag ${module}_version)

    ExternalProject_Add(
        ${module}
        PREFIX ${module}
        GIT_REPOSITORY ${${module}_url}
        GIT_TAG ${${module}_tag}
        SOURCE_DIR "${external_download_dir}/${module}"
        BUILD_IN_SOURCE 0
        BUILD_COMMAND make -j 8
        INSTALL_COMMAND make install
        CMAKE_CACHE_ARGS
            -DCMAKE_INSTALL_PREFIX:STRING=${external_install_path}
    )

    # make sure the dir exist
    set(include_dir ${external_install_path}/include/eigen3)
    touch_fold(include_dir)

    add_library(${${module}_target_name} UNKNOWN IMPORTED GLOBAL)
    set_target_properties(
        ${${module}_target_name}
        PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${include_dir}"
    )
    add_dependencies(${${module}_target_name} ${module})
    unset(include_dir)
endif()