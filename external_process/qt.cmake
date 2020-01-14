include(popular_message)
include(ProjectBaseSet)
cmakelists_base_header()
project_base_system_message()
include(external_setting)
include(file_operation)

set(module Qt)
set(qt_target_name Qt::Qt)

set(
    ${module}_supported_version 
    5.9.9
    )

set(
    ${module}_supported_url
    http://download.qt.io/archive/qt/5.9/5.9.9/single/qt-everywhere-opensource-src-5.9.9.tar.xz
    )

set(
    ${module}_supported_hash
    5ce285209290a157d7f42ec8eb22bf3f1d76f2e03a95fc0b99b553391be01642
    )

version_selector(${module} ${module}_supported_version 5.9.9)
version_url_hash_match(${module} ${module}_supported_version ${module}_supported_url ${module}_supported_hash ${module}_version)

# to get the gfalgs_build_type
default_external_project_build_type(${module})

# to get the _gflags_build_type shared or static
project_build_shared(${module})

find_package(${module} ${${module}_version} CONFIG NO_CMAKE_PACKAGE_REGISTRY PATHS ${external_install_path})

macro(build_and_install_qt)
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${external_download_dir})
    get_file_name_from_url(file_name ${module}_url)
    message(STATUS ${file_name})
    message(STATUS ${${module}_url})
    message(STATUS ${external_download_dir})
    touch_file(touch_${module} module ${module}_url ${module}_hash external_download_dir file_name)
    add_custom_target(
        ${module}_install
        COMMAND ${CMAKE_COMMAND} -E ${external_download_dir} ls
        )
    add_dependencies(${module}_install ${touch_${module}})
    add_library(_${module} INTERFACE IMPORTED)
    set_target_properties(
        _${module}
        PROPERTIES
        INTERFACE_LINK_LIBRARIES "${module}_install"
    )
    add_dependencies(_${module} ${module}_install)
endmacro(build_and_install_qt)

if(${${module}_FOUND})
else()
    build_and_install_qt()
    add_library(${qt_target_name} INTERFACE IMPORTED)
    set_target_properties(
        ${qt_target_name}
        PROPERTIES
        INTERFACE_LINK_LIBRARIES "_${module}"
    )
    add_dependencies(${qt_target_name} _${module})
endif()