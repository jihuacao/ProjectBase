include(popular_message)
include(external_setting)
cmakelists_base_header()
project_base_system_message()
include(ExternalProject)
include(file_read_method)
include(file_operation)
include(fold_operation)
include(ProjectBaseSet)
# these could be remove
set(module boost)
set(external_project_add_target _boost)
set(url_label_name boost)
set(extension tar.gz)
set(component_lib_prefix Boost)
set(generate_boost_op_name _${module})
# this should be retain
set(generate_boost_imported_name boost_generate_target)

# generate a boost component config file to the CMAKE_SOURCE_DIR
## find the config file
find_file(component_config_file boost_component_config NO_DEFAULT_PATH PATHS ${CMAKE_SOURCE_DIR})
if(${component_config_file} STREQUAL "component_config_file-NOTFOUND")
    message(STATUS "copy boost_component_config to ${CMAKE_SOURCE_DIR}")
    file(COPY ${ProjectBase_boost_component_config_file_rpath} DESTINATION ${CMAKE_SOURCE_DIR})
else()
    message(STATUS found:${component_config_file})
endif()
set(component_file_rpath ${CMAKE_SOURCE_DIR}/${ProjectBase_boost_component_config_file_name})
execute_process(
    COMMAND 
    python
    ${ProjectBase_script_component_config_file_fix} 
    --using_component_file=${component_file_rpath}
    --standard_component_file=${ProjectBase_boost_component_config_file_rpath}
    )
component_config_read(component_file_rpath module)
message(STATUS "with boost component:")
foreach(with ${${module}_with})
   message(STATUS :${with}) 
endforeach()

message(STATUS "without boost component:")
foreach(without ${${module}_without})
   message(STATUS :${without}) 
endforeach()

list(
    APPEND 
    ${module}_all_version 
    "1.71.0"
    )

set(
    ${module}_supported_url 
    https://dl.bintray.com/${url_label_name}org/release/1.71.0/source/${url_label_name}_1_71_0.tar.gz
    )

set(
    ${module}_supported_hash
    96b34f7468f26a141f6020efb813f1a2f3dfb9797ecf76a7d7cbd843cc95f5bd
    )

version_selector(${module} ${module}_all_version "1.71.0")

default_external_project_build_type(${module})

project_build_shared(${module})

set(Boost_USE_DEBUG_LIBS OFF)
set(Boost_USE_RELEASE_LIBS OFF)
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_STATIC_RUNTIME OFF)
set(Boost_USE_DEBUG_RUNTIME OFF)
set(Boost_COMPILER OFF)
set(Boost_PYTHON_VERSION OFF)
set(Boost_VERBOSE OFF)
set(Boost_DEBUG OFF)
if(${_${module}_build_type} STREQUAL "RELEASE")
    set(Boost_USE_RELEASE_LIBS ON)
else()
    set(Boost_USE_DEBUG_LIBS ON)
endif()
if(${_${module}_build_shared})
    set(Boost_USE_STATIC_LIBS OFF)
else()
    set(Boost_USE_STATIC_LIBS ON)
endif()
message(STATUS "Prefix Configuration:
````````Boost_USE_DEBUG_LIBS: ${Boost_USE_DEBUG_LIBS}
````````Boost_USE_RELEASE_LIBS: ${Boost_USE_RELEASE_LIBS}
````````Boost_USE_STATIC_LIBS: ${Boost_USE_STATIC_LIBS}
````````Boost_USE_STATIC_RUNTIME: ${Boost_USE_STATIC_RUNTIME}
````````Boost_USE_DEBUG_RUNTIME: ${Boost_USE_DEBUG_RUNTIME}
````````Boost_COMPILER: ${Boost_COMPILER}
````````Boost_PYTHONT_VERSION: ${Boost_PYTHON_VERSION}
````````Boost_VERBOSE: ${Boost_VERBOSE}
````````Boost_DEBUG: ${Boost_DEBUG}")

find_package(Boost ${${module}_version} COMPONENTS ${${module}_with} CONFIG NO_CMAKE_PACKAGE_REGISTRY PATHS ${external_install_path})

function(download_and_build_boost)
    version_url_hash_match(${module} ${module}_all_version ${module}_supported_url ${module}_supported_hash ${module}_version)
    
    string(REPLACE "." "_" url_version_name ${${module}_version})
    string(CONCAT download_file_name ${url_label_name}_ ${url_version_name}.${extension})

    touch_file(target module ${module}_url ${module}_hash external_download_dir download_file_name)

    # get the component option
    list(LENGTH ${module}_with component_amount)
    if(component_amount EQUAL 0)
        message(WARNING "with ${component_amount} component for ${module}")
    else()
    endif()
    set(with_component_option)
    foreach(component ${${module}_with})
       set(with_component_option ${component},${with_component_option})
    endforeach()
    message(STATUS "with_component_option: ${with_component_option}")
    if(${_${module}_build_type} STREQUAL "RELEASE")
        set(_variant release)
    else()
        set(_variant debug)
    endif()
    if(${_${module}_build_shared})
        set(_link shared)
    else()
        set(_link static)
    endif()
    
    add_custom_target(${generate_boost_op_name}
        COMMAND ${CMAKE_COMMAND} -E chdir ${external_download_dir} tar -xvf ${download_file_name} >./decompression_info.txt
        COMMAND ${CMAKE_COMMAND} -E chdir ${external_download_dir}/${url_label_name}_${url_version_name} ./bootstrap.sh --with-libraries=${with_component_option} --prefix=${external_install_path} variant=_variant link=_link runtime-link=shared threading=multi > bootstrap_info.txt
        COMMAND ${CMAKE_COMMAND} -E chdir ${external_download_dir}/${url_label_name}_${url_version_name} ./b2 > b2_info.txt
        COMMAND ${CMAKE_COMMAND} -E chdir ${external_download_dir}/${url_label_name}_${url_version_name} ./b2 install --prefix=${external_install_path} > b2_install_info.txt
    )
    add_dependencies(${generate_boost_op_name} ${target})
    unset(with_component_option)
    unset(_variant)
    unset(_link)
endfunction(download_and_build_boost)

function(build_boost_target version components shared _build_type install_path)
    if(${${shared}})
        set(posefix so)
    else()
        set(posefix a)
    endif()
    if(${${_build_type}} STREQUAL "RELEASE")
        set(type )
    else()
        set(type d)
    endif()

    set(include_dir ${${install_path}}/include)
    set(lib_dir ${${install_path}}/lib)
    # make sure the directory exists
    touch_fold(include_dir)
    touch_fold(lib_dir)

    foreach(component ${${components}})
        add_library(${module}_${component} UNKNOWN IMPORTED)
        set_target_properties(${module}_${component}
            PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${${install_path}}/include"
            INTERFACE_LINK_DIRECTORIES "${${install_path}}/lib"
            IMPORTED_LOCATION_${_${module}_build_type} "${${install_path}}/lib/libboost_${component}${type}.${posefix}"
        )
        list(APPEND libraries ${module}_${component};)
    endforeach()
    # importent: waiting to write the note
    add_library(${generate_boost_imported_name} INTERFACE IMPORTED)
    set_target_properties(${generate_boost_imported_name}
        PROPERTIES
        INTERFACE_LINK_LIBRARIES "${libraries}"
    )
    add_dependencies(${generate_boost_imported_name} ${generate_boost_op_name})
    unset(include_dir)
    unset(lib_dir)
endfunction(build_boost_target)

macro(add_total_boost_component_link)
    if(${module}_total_found)
        set(${module}_target_name Boost::boost)
    else()
        set(${module}_target_name ${generate_boost_imported_name})
    endif()
    message(STATUS ${${module}_target_name})
endmacro(add_total_boost_component_link)

if(Boost_FOUND)
    message(DEBUG "BOOST FOUND(TEMP)")
    set(component_found)
    set(component_not_found)
    foreach(component ${${module}_with})
        if(${module}_${component}_FOUND)
            list(APPEND component_found ${component})
        else()
            list(APPEND component_not_found ${component})
        endif()
    endforeach()
    list(LENGTH component_not_found not_found_amount)
    if(${not_found_amount} EQUAL 0)
        set(${module}_total_found ON)
    else()
        set(${module}_total_found OFF)
    endif()
    message(STATUS "${module} found component: ${component_found}; not found component: ${component_not_found}")
    if(${module}_total_found)
        message(DEBUG "TOTAL FOUND(TEMP)")
        add_total_boost_component_link()
    else()
        message(DEBUG "NOT TOTAL FOUND(TEMP)")
        download_and_build_boost()
        build_boost_target(${module}_version ${module}_with _${module}_build_shared _${module}_build_type external_install_path)
        add_total_boost_component_link()
    endif()
else()
    message(DEBUG "BOOST NOT FOUND(TEMP)")
    download_and_build_boost()
    build_boost_target(${module}_version ${module}_with _${module}_build_shared _${module}_build_type external_install_path)
    add_total_boost_component_link()
endif()

unset(component_found)
unset(component_not_found)
unset(${module}_total_found)
unset(module)
unset(external_project_add_target)
unset(url_label_name)
unset(extension)
unset(component_lib_prefix)
unset(generate_boost_op_name)