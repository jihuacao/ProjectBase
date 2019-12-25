set(external_install_path ${CMAKE_CURRENT_BINARY_DIR}/install CACHE STRING "the path for installing the external")
option(external_build_shared "the lib link type for building the external" ON)
set(external_download_dir ${CMAKE_CURRENT_BINARY_DIR} CACHE STRING "the dir for download the source of the third party repo")

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
                list(FIND ${supported_tag} ${matched_index} matched_tag)
                message(DEPRECATION "find tag ${matched_tag} for ${${project}}")
                set(${project}_tag ${matched_tag})
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
