# 消息等级:FATAL_ERROR WARNING STATUS DEBUG...
function(external_cmake_args)
    #[[
        cache更新，然后再更新非cache变量，因为cache往往是传入相对路径，
        而使用中的变量最好是绝对路径
    ]]

    #[[ 获取路径：可用与转化相对路径为绝对路径，
    获取路径的相关集合操作，
    注意这里是通过相对路径获取绝对路径，
    注意指定父路径需要使用BASE_DIR <dir>]]
    # 
    set(var_external_root_dir ${CMAKE_CURRENT_SOURCE_DIR}/external CACHE STRING "the absolute dir to install the external, You Should Specify An Absolute Dir")
    set(external_root_dir ${var_external_root_dir})
    # get_filename_component(external_root_dir ${var_external_root_dir} ABSOLUTE BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
    message(STATUS "external_root_dir: ${external_root_dir}")
    set(external_root_dir ${external_root_dir} PARENT_SCOPE)

    # todo: 逐步替换external_install_path为external_install_dir
    set(var_external_install_dir ${external_root_dir}/external_install CACHE STRING "the dir to install the external")
    set(external_install_dir ${var_external_install_dir})
    message(STATUS "external_install_dir: ${external_install_dir}")
    set(external_install_dir ${external_install_dir} PARENT_SCOPE)
    set(external_install_path ${external_install_dir} PARENT_SCOPE)

    option(external_build_shared "the lib link type for building the external" ON)
    message(STATUS "external_build_shared: ${external_build_shared}")

    set(var_external_download_dir ${external_root_dir}/external_download CACHE STRING "the dir to store the external")
    set(external_download_dir ${var_external_download_dir})
    # get_filename_component(external_download_dir ${var_external_download_dir} ABSOLUTE BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
    message(STATUS "external_download_dir: ${external_download_dir}")
    set(external_download_dir ${external_download_dir} PARENT_SCOPE)

    set(var_external_build_dir ${external_root_dir}/external_build CACHE STRING "the dir to build the external project")
    # get_filename_component(external_build_dir ${var_external_build_dir}  ABSOLUTE BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
    set(external_build_dir ${var_external_build_dir})
    message(STATUS "external_build_dir: ${external_build_dir}")
    set(external_build_dir ${external_build_dir} PARENT_SCOPE)
endfunction(external_cmake_args)
##################################################################
# create a external_dependencies property in CACHED_VARIABLE
# this property 
##################################################################
define_property(CACHED_VARIABLE PROPERTY external_dependencies BRIEF_DOCS asdasd FULL_DOCS asd)

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
        string(TOUPPER ${CMAKE_BUILD_TYPE} _${project}_build_type)
    elseif("${${project}_build_type}" IN_LIST ${supported_build_type})
        string(TOUPPER ${${project}_build_type} _${project}_build_type)
    else()
        message(FATAL_ERROR "unsupported gtest_build_type: ${${project}_build_type}")
    endif()
endmacro(external_project_build_type)

macro(default_external_project_build_type project)
    list(APPEND supported_build_type "FOLLOW_CMAKE_BUILD_TYPE" "Release" "Debug")
    set(default_build_type "FOLLOW_CMAKE_BUILD_TYPE")
    external_project_build_type(${project} supported_build_type default_build_type)
endmacro(default_external_project_build_type)

function(default_external_project_install_dir module)
    set(gtest_install_dir ${external_install_dir} CACHE STRING "${module} install dir")
endfunction(default_external_project_install_dir)

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

##################################################################
# this function extract the configure which contain the include_dirs 
# link_dirs link_name etc.
# this function is dependen on the external_process module, in which
# create the a cache-property for getting the configure
# param:
#   targets: list contains any existed external target
#   to_target: should be an existed target
# result:
#   link_dirs:
#   sonames:
#   include_dirs:
#   predefines:
##################################################################
function(default_external_config_getter target to_target)
    get_target_property(${target}_include_dirs ${target} INTERFACE_INCLUDE_DIRECTORIES) 
    set_property(TARGET ${${to_target}} APPEND PROPERTY INCLUDE_DIRECTORIES ${${target}_include_dirs})
    get_target_property(${target}_link_library ${target} IMPORTED_SN)
endfunction(default_external_config_getter)


macro(external_configs_getter targets to_target)
    foreach(target IN LISTS ${targets})
        message(STATUS ${target})
        eval("${target}_get_config" to_target)
    endforeach()
endmacro(external_configs_getter)

#[[
    解析外部依赖构建函数的参数
    * common_args
        * options
            * EXTERNAL_EXPORT 
                * 如果被设置，那么外部依赖将会被当做向外暴露的第三方库
                * 如果没设置，那么该依赖会被当做内部依赖
]]
function(configure_git_args)
endfunction(configure_git_args)
macro(configure_common_args)
    set(external_options
        EXTERNAL_EXPORT
        ${external_options}
    )
    set(external_one_value_args
        ${external_one_value_args}
    )
    set(external_multi_value_args
        ${external_multi_value_args}
    )
endmacro(configure_common_args)
macro(parser_the_arguments external_project_name external_project_type)
    configure_common_args()
    if(${external_project_type} MATCHES "git")
        configure_git_args()
    else()
        message(FATAL "Not Support this TYPE: ${type}")
    endif()
    cmake_parse_arguments(${external_project_name} "${external_options}" "${external_one_value_args}" "${external_multi_value_args}" ${ARGN})
endmacro(parser_the_arguments)

#[[
    cmake_external_project_common_args
    @param[in] external_project_name 传入标志名称
]]
function(cmake_external_project_common_args external_project_name)
    string(TOLOWER ${CMAKE_BUILD_TYPE} lower_cmake_build_type)
    string(TOLOWER ${CMAKE_GENERATOR_PLATFORM} lower_cmake_generator_platform)
    macro(do_message msg)
        message(DEBUG 
        "${external_project_name} ExternalProject_Add::${msg}")
    endmacro(do_message)
    macro(set_args_variable option var)
        string(TOUPPER ${option} upper_option)
        do_message(
        "${upper_option}(${external_project_name}_${option})=${var}")
        set(${external_project_name}_${option} ${var} PARENT_SCOPE)
    endmacro(set_args_variable)
    macro(do_cmake_message msg)
        message(DEBUG 
        "${external_project_name} ExternalProject_Add::${msg}")
    endmacro(do_cmake_message)
    macro(set_cmake_args_variable option var)
        string(TOUPPER ${option} upper_option)
        do_cmake_message(
        "${upper_option}(${external_project_name}_${option})=${var}")
        set(${external_project_name}_${option} ${var} PARENT_SCOPE)
    endmacro(set_cmake_args_variable)
     
    set_args_variable(build_in_source OFF)
    set_args_variable(source_dir ${external_download_dir}/${external_project_name})
    #[[
        :generate the build dir
        * visual studio可以多种configuration共存于同一build dir中，**但是还是放到不同位置吧**
        * visual studio可以多种configuration编译中间文件相互独立
        * **但有些project不对不同configuration的output name进行处理，因此不能install到同一dir中**
        * makefiles类型不能多种configuration共存，build_dir install_dir跟随不同的configuration platform变化
        * ${${external_project_name}_EXTERNAL_EXPORT}为True安装到public位置，为False安装到internal位置
    ]]
    if(${CMAKE_GENERATOR} MATCHES "Makefiles")
        set_cmake_args_variable(cmake_generator ${CMAKE_GENERATOR})
        set_args_variable(binary_dir ${external_build_dir}/${external_project_name}_${lower_cmake_generator_platform}_${lower_cmake_build_type})
    elseif(${CMAKE_GENERATOR} MATCHES "Visual Studio")
        set_cmake_args_variable(cmake_generator ${CMAKE_GENERATOR})
        set_args_variable(binary_dir ${external_build_dir}/${external_project_name}_${lower_cmake_generator_platform})
        if(${${external_project_name}_EXTERNAL_EXPORT})
            set_cmake_args_variable(cmake_install_prefix ${external_install_dir}/${lower_cmake_build_type}_public)
        else()
            set_cmake_args_variable(cmake_install_prefix ${external_install_dir}/${lower_cmake_build_type}_internal)
        endif()
    else()
        message(FATAL "${external_project_name} UNKNOWN CMAKE_GENERATOR")
    endif()
    #[[
        todo: configuration the build command
        do not need build command，because the external project is cmake project.
        using cmake -build as default.
        different build command for different generator would be call.
    ]]
    set_args_variable(build_commnad "")
    #[[
        todo: configure the install command
        * install or not is depend on the target link_library type
            * PUBLIC INTERFACE: need install
            * PRIVATE: do not need install
    ]]
    if(${${external_project_name}_build_shared})
        set_args_variable(install_command "")
    else()
        set_args_variable(install_command "")
    endif()
endfunction(cmake_external_project_common_args)