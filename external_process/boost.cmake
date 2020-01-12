include(popular_message)
include(external_setting)
cmakelists_base_header()
project_base_system_message()
include(ExternalProject)
include(file_read_method)
set(module Boost)
set(external_project_add_target _boost)

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
    https://dl.bintray.com/${module}org/release/1.71.0/source/${module}_1_71_0.tar.gz
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

find_package(${module} "1.71" COMPONENTS ${${module}_with} CONFIG HINTS /home/sins/Download/boost_1_71_0/install/)

function(download_and_build_boost)
    version_url_hash_match(${module} ${module}_all_version ${module}_supported_url ${module}_supported_hash ${module}_version)

    string(REPLACE "." "_" boost_download_file_name boost.${${module}_version})
    find_file(found_download_file ${boost_download_file_name} ${external_download_path})
    set(actual_download_command axel -k -n 10 -av ${${module}_url} -o ${external_download_dir}/${boost_download_file_name})
    if(${found_download_file} STREQUAL "found_download_file-NOTFOUND")
        set(download_command ${actual_download_command})
    else()
        file(SHA256 ${external_download_dir}/${boost_download_file_name} exist_sha)
        if(${exist_sha} STREQUAL ${${module}_hash})
            set(download_command )
        else()
            set(download_command ${actual_download_command})
        endif()
    endif()
    unset(actual_download_command)
    add_custom_command(OUTPUT a COMMAND echo asd COMMAND echo aswf)
    message(FATAL_ERROR "ad")

    # mkdir the 
    add_custom_command()
    ExternalProject_Add(
        _boost
        PREFIX ${module}-${${module}_version}
        URL ${${module}_url}
        URL_HASH:SHA256="${${module}_hash}"
        DOWNLOAD_COMMAND ${download_command}
        DOWNLOAD_DIR "${external_download_dir}"
        UPDATE_COMMAND ""
    )
    unset(boost_download_file_name)
    unset(found_download_file)
    unset(download_command)
endfunction(download_and_build_boost)

macro(add_total_boost_component_link)
    add_library(${module}_total UNKNOWN IMPORTED)
    foreach(component ${${module}_with})
        set_property(TARGET ${module}_total APPEND PROPERTY INTERFACE_LINK_INTERFACE_LIBRARIES_RELEASE "${module}::${component}")
    endforeach()
    get_target_property(t ${module}_total INTERFACE_LINK_INTERFACE_LIBRARIES_RELEASE)
    message(STATUS "${t}")
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
