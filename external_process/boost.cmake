include(popular_message)
include(external_setting)
cmakelists_base_header()
project_base_system_message()
include(ExternalProject)
include(file_read_method)
include(file_operation)
set(module Boost)
set(external_project_add_target _boost)
set(url_label_name boost)
set(extension tar.gz)
set(component_lib_prefix Boost)

# generate a boost component config file to the CMAKE_SOURCE_DIR
## find the config file
find_file(component_config_file boost_component_config NO_DEFAULT_PATH PATHS ${CMAKE_SOURCE_DIR})
if(${component_config_file} STREQUAL "component_config_file-NOTFOUND")
    message(STATUS copy boost_component_config to ${CMAKE_SOURCE_DIR})
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

set(${module}_USE_DEBUG_LIBS OFF)
set(${module}_USE_RELEASE_LIBS OFF)
set(${module}_USE_STATIC_LIBS OFF)
set(${module}_USE_STATIC_RUNTIME OFF)
set(${module}_USE_DEBUG_RUNTIME OFF)
set(${module}_COMPILER OFF)
set(${module}_PYTHON_VERSION OFF)
set(${module}_VERBOSE OFF)
set(${module}_DEBUG OFF)
if(${_${module}_build_type} STREQUAL "RELEASE")
    set(${module}_USE_RELEASE_LIBS ON)
else()
    set(${module}_USE_DEBUG_LIBS ON)
endif()
if(${_${module}_build_shared})
    set(${module}_USE_STATIC_LIBS OFF)
else()
    set(${module}_USE_STATIC_LIBS ON)
endif()
message(STATUS "Prefix Configuration:
````````${module}_USE_DEBUG_LIBS: ${${module}_USE_DEBUG_LIBS}
````````${module}_USE_RELEASE_LIBS: ${${module}_USE_RELEASE_LIBS}
````````${module}_USE_STATIC_LIBS: ${${module}_USE_STATIC_LIBS}
````````${module}_USE_STATIC_RUNTIME: ${${module}_USE_STATIC_RUNTIME}
````````${module}_USE_DEBUG_RUNTIME: ${${module}_USE_DEBUG_RUNTIME}
````````${module}_COMPILER: ${${module}_COMPILER}
````````${module}_PYTHONT_VERSION: ${${module}_PYTHON_VERSION}
````````${module}_VERBOSE: ${${module}_VERBOSE}
````````${module}_DEBUG: ${${module}_DEBUG}")

find_package(${module} "1.71" COMPONENTS ${${module}_with} CONFIG HINTS ${external_install_path})

function(download_and_build_boost)
    version_url_hash_match(${module} ${module}_all_version ${module}_supported_url ${module}_supported_hash ${module}_version)
    
    string(REPLACE "." "_" url_version_name ${${module}_version})
    string(CONCAT download_file_name ${url_label_name}_ ${url_version_name}.${extension})

    touch_file(target module ${module}_url ${module}_hash external_download_dir download_file_name)

    # get the component option
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
    
    add_custom_target(_${module}
        COMMAND ${CMAKE_COMMAND} -E chdir ${external_download_dir} tar -xvf ${download_file_name} >./decompression_info.txt
        COMMAND ${CMAKE_COMMAND} -E chdir ${external_download_dir}/${url_label_name}_${url_version_name} ./bootstrap.sh --with-libraries=${with_component_option} --prefix=${external_install_path} variant=_variant link=_link runtime-link=shared threading=multi
        COMMAND ${CMAKE_COMMAND} -E chdir ${external_download_dir}/${url_label_name}_${url_version_name} ./b2
        COMMAND ${CMAKE_COMMAND} -E chdir ${external_download_dir}/${url_label_name}_${url_version_name} ./b2 install --prefix=${external_install_path}
    )
    add_dependencies(_${module} ${target})
    unset(with_component_option)
    unset(_variant)
    unset(_link)
endfunction(download_and_build_boost)

function(build_boost_target version components install_path)
    foreach(component ${components})
        add_library(${component_lib_prefix}::${component} UNKNOW IMPORTED)
        set_target_properties(
            ${component_lib_prefix}::${component}
            PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${${install_path}}/include"
            INTERFACE_COMPILE_DEFINITIONS "BOOST_ALL_NO_LIB"
        )
    endforeach()
endfunction(build_boost_target)

macro(add_total_boost_component_link)
    add_library(${module}_total UNKNOWN IMPORTED)
    foreach(component ${${module}_with})
        #set_property(TARGET ${module}_total APPEND PROPERTY IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "${module}::${component}")
        add_dependencies(${module}_total ${module}::${component})
    endforeach()
    show_target_properties(Boost::filesystem)
    set_target_properties(${module}_total
        PROPERTIES 
        INTERFACE_LINK_LIBRARIES "Boost::filesystem"
    )
    set(${module}_target_name ${module}_total)
    message(STATUS ${${module}_target_name})
endmacro(add_total_boost_component_link)

if(${module}_FOUND)
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
        add_total_boost_component_link()
    else()
        download_and_build_boost()
        # todo: make the component libraries
        add_total_boost_component_link()
    endif()
else()
    download_and_build_boost()
    # todo: make the component libraries
    add_total_boost_component_link()
endif()
unset(component_found)
unset(component_not_found)
unset(${module}_total_found)
