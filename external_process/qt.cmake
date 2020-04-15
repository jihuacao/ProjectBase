include(popular_message)
include(ProjectBaseSet)
cmakelists_base_header()
project_base_system_message()
include(external_setting)
include(file_read_method)
include(file_operation)

set(module Qt)
set(download_file_extension tar.xz)
# this should be retain
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

string(REPLACE "." ";" ${module}_split_version ${${module}_version})
list(GET ${module}_split_version 0 ${module}_major_version)
set(${module}_target_package ${module}${${module}_major_version})
message(STATUS "the ${module}_target_package: ${${module}_target_package}")

find_file(component_config_file qt${${module}_major_version}_component_config NO_DEFAULT_PATH PATHS ${CMAKE_SOURCE_DIR})
if(${component_config_file} STREQUAL "component_config_file-NOTFOUND")
    message(STATUS "copt qt_component_config to ${CMAKE_SOURCE_DIR}")
    file(COPY ${ProjectBase_qt${${module}_major_version}_component_config_file_rpath} DESTINATION ${CMAKE_SOURCE_DIR})
else()
    message(STATUS "found:${component_config_file}")
endif()
set(component_file_rpath ${CMAKE_SOURCE_DIR}/${ProjectBase_qt${${module}_major_version}_component_config_file_name})
execute_process(
    COMMAND
    python 
    ${ProjectBase_script_component_config_file_fix}
    --using_component_file=${component_file_rpath}
    --standard_component_file=${ProjectBase_qt${${module}_major_version}_component_config_file_rpath}
)
component_config_read(component_file_rpath module)
message(STATUS "with qt${${module}_major_version} component:")
foreach(with ${${module}_with})
    message(STATUS ":${with}")
endforeach()
message(STATUS "without qt${${module}_major_version} component:")
foreach(without ${${module}_without})
    message(STATUS ":${without}")
endforeach()

find_package(${${module}_target_package} ${${module}_version} COMPONENTS ${${module}_with} CONFIG NO_CMAKE_PACKAGE_REGISTRY PATHS ${external_install_path})

macro(build_and_install_qt)
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${external_download_dir})
    get_file_name_from_url(download_file_name ${module}_url)
    touch_file(touch_${module} module ${module}_url ${module}_hash external_download_dir download_file_name)

    string(REPLACE ".${download_file_extension}" "" fold_name ${download_file_name})
    message(STATUS ${fold_name})
    if(${_${module}_build_type} STREQUAL "RELEASE")
        set(build_type_options -release)
    else()
        set(build_type_options -debug)
    endif()
    if(${_${module}_build_shared})
        set(link_type_options -shared)
    else()
        set(link_type_options -static)
    endif()
    if(NOT qt_custom_options)
        set(qt_custom_options)
        message(STATUS "the qt configure custom options is : ${qt_custom_set_options}
        you can set qt_custom_set_options before call this mudule")
    endif()
    message(STATUS "the qt_custom_options:${qt_custom_options}")
    add_custom_target(
        ${module}_install
        #COMMAND ${CMAKE_COMMAND} -E chdir ${external_download_dir} tar -xvf ${external_download_dir}/${download_file_name} > decompression_info.txt
        COMMAND ${CMAKE_COMMAND} -E chdir ${external_download_dir}/${fold_name} ./configure
            #-make libs tools examples
            -silent
            -prefix ${external_install_path}
            ${build_type_options}
            ${link_type_options}
            ${qt_custom_options} > configure_info.txt
        COMMAND ${CMAKE_COMMAND} -E chdir ${external_download_dir}/${fold_name} make -j$(nproc)
        COMMAND ${CMAKE_COMMAND} -E chdir ${external_download_dir}/${fold_name} make install
        )
    add_dependencies(${module}_install ${touch_${module}})
    add_library(_${module} INTERFACE IMPORTED)
    set_target_properties(
        _${module}
        PROPERTIES
        INTERFACE_LINK_LIBRARIES "${module}_install"
    )
    add_dependencies(_${module} ${module}_install)
    unset(build_type_options)
    unset(link_type_options)
    unset(fold_name)
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
unset(${module}_split_version)
unset(${module}_major_version)
unset(component_config_file)