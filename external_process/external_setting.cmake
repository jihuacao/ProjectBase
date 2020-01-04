set(external_install_path ${CMAKE_CURRENT_BINARY_DIR}/install CACHE STRING "the path for installing the external")
message(STATUS "external_install_path: ${external_install_path}")
option(external_build_shared "the lib link type for building the external" ON)
message(STATUS "external_build_shared: ${external_build_shared}")
set(external_download_dir ${CMAKE_SOURCE_DIR}/external CACHE STRING "the dir for download the source of the third party repo")
message(STATUS "external_download_dir: ${external_download_dir}")

##################################################################
# macro
##################################################################
macro(project_build_shared project)
    # to get the _gtest_build_type shared or static
    set(${project}_build_shared ${external_build_shared} CACHE STRING "specifical build the gtest in shared or follow in external_build_shared")
    set_property(CACHE ${project}_build_shared PROPERTY STRINGS "ON" "OFF" "FOLLOW_EXTERNAL_BUILD_SHARED")
    if("${${project}_build_shared}" STREQUAL "FOLLOW_EXTERNAL_BUILD_SHARED")
        set(_${project}_build_shared ${external_build_shared})
    else()
        set(_${project}_build_shared ${${project}_build_shared})
    endif()
endmacro(project_build_shared)

##################################################################
# this macro set the project version selected variable tab
# project: the name of the project
# all_version: the versions which the process supported and can be choose
# default_version: the default version which we supported
#
# what we can get after this macro: we get the variable ${project}_version
##################################################################
macro(version_selector project all_version default_version)
    set(${project}_all_version ${${all_version}})
    set(${project}_version ${default_version} CACHE STRING "${project} version")
    set_property(CACHE ${project}_version PROPERTY STRINGS ${${project}_all_version})
endmacro(version_selector)

##################################################################
# this macro find the matched tag for the project which is in the git
# project: the name of the project
# supported_version: the versions which the process supported or the project has
# supported_tag: the tags which corresponding to the supported_version
# version: the version which user choose for the project
#
# what we can get after this macro: we get the variable ${project}_tag
##################################################################
macro(version_tag_matcher project supported_version supported_tag version)
    if(${${version}} IN_LIST ${supported_version})
        list(LENGTH ${supported_version} ${project}_supported_version_length)
        list(LENGTH ${supported_tag} ${project}_supported_tag_length)
        if(${${project}_supported_version_length} EQUAL ${${project}_supported_tag_length})
            list(FIND ${supported_version} ${${version}} matched_index)
            if(NOT(${matched_index} EQUAL -1))
                list(GET ${supported_tag} ${matched_index} matched_tag)
                if(NOT(matched_tag EQUAL -1))
                    message(STATUS "find tag ${matched_tag} for ${project}")
                    set(${project}_tag ${matched_tag})
                else()
                    message(FATAL_ERROR "index: ${matched_index} not in supported_tag: ${${supported_tag}}")
                endif()
            else()
                message(FATAL_ERROR "can not find ${${version}} in supported_version: ${${supported_version}}")
            endif()
        else()
            message(FATAL_ERROR "${${project}} supported_version: ${${supported_version}} vs. supported_tag: ${${supported_tag}} , should be in same length")
        endif()
    else()
        message(FATAL_ERROR "version of ${${project}}: ${${version}} is not in the supported_version: ${${supported_version}}")
    endif()
endmacro(version_tag_matcher)

##################################################################
# this macro
##################################################################
macro(external_project_build_type project supported_build_type default_build_type)
    list(FIND ${supported_build_type} ${${default_build_type}} index)
    if(${index} EQUAL -1)
        message(FATAL_ERROR "default_build_type: ${${default_build_type}} is not in the supported_build_type: ${${supported_build_type}}")
    endif()
    set(${project}_build_type ${${default_build_type}} CACHE STRING "the specifical option for gtest, if the gtest_build_type is set")
    set_property(CACHE ${project}_build_type PROPERTY STRINGS ${${supported_build_type}})
    if("${${project}_build_type}" STREQUAL "FOLLOW_CMAKE_BUILD_TYPE")
        set(_${project}_build_type ${CMAKE_BUILD_TYPE})
    elseif("${${project}_build_type}" IN_LIST ${supported_build_type})
        set(_${project}_build_type ${${project}_build_type})
    else()
        message(FATAL_ERROR "unsupported gtest_build_type: ${${project}_build_type}")
    endif()
endmacro(external_project_build_type)

macro(default_external_project_build_type project)
    list(APPEND supported_build_type "FOLLOW_CMAKE_BUILD_TYPE" "Release" "Debug")
    set(default_build_type "FOLLOW_CMAKE_BUILD_TYPE")
    external_project_build_type(${project} supported_build_type default_build_type)
endmacro(default_external_project_build_type)


##################################################################
# this macro find the matched url and hash for the version
# project: the module name
# supported_version: the list which contains the supported version string
# supported_url: the list which contains the supported url, corresponding to the supported version
# supported_hash: the list which contains the supported hash, corresponding to the supported version
##################################################################
macro(version_url_hash_match project supported_version supported_url supported_hash version)
    if(${${version}} IN_LIST ${supported_version})
        list(LENGTH ${supported_version} ${project}_supported_version_length)
        list(LENGTH ${supported_url} ${project}_supported_url_length)
        list(LENGTH ${supported_hash} ${project}_supported_hash_length)
        if((${${project}_supported_version_length} EQUAL ${${project}_supported_url_length}) AND (${${project}_supported_version_length} EQUAL ${${project}_supported_hash_length}))
            list(FIND ${supported_version} ${${version}} matched_index)
            if(NOT(${matched_index} EQUAL -1))
                list(GET ${supported_url} ${matched_index} matched_url)
                list(GET ${supported_hash} ${matched_index} matched_hash)
                if((NOT(matched_url EQUAL -1)) AND (NOT(matched_hash EQUAL -1)))
                    message(STATUS "find url ${matched_url} and hash ${matched_hash} for ${project}")
                    set(${project}_url ${matched_url})
                    set(${project}_hash ${matched_hash})
                else()
                    message(FATAL_ERROR "index: ${matched_index} not in supported_url: ${${supported_url}}")
                endif()
            else()
                message(FATAL_ERROR "can not find ${${version}} in supported_version: ${${supported_version}}")
            endif()
        else()
            message(FATAL_ERROR "${${project}} supported_version: ${${supported_version}} vs. supported_url: ${${supported_url}} vs. supported_hash: ${${supported_hash}}, should be in same length")
        endif()
    else()
        message(FATAL_ERROR "version of ${${project}}: ${${version}} is not in the supported_version: ${${supported_version}}")
    endif()
endmacro(version_url_hash_match)

# ##################################################################
# # this macro find the target package and check the version of the found target 
# # is in the supported version or not
# # target_name: the name of the target package ,use in the find_package
# # supported_version: the versions which is fit to the project
# ##################################################################
# macro(check_env_package target_name supported_version)
#     find_package(${${target_name}})
# endmacro(check_env_package)