##@brief
# @note
# @param[in]
# @param[in]
# @return 
# @time 2023-07-07
# @author cjh
macro(add_base_module module_name)
    ## compile method 
    include(popular_message)
    cmakelists_base_header()
    aux_source_directory(${CMAKE_CURRENT_LIST_DIR} DIR_LIB_SRCS)
    message(STATUS "srcs: ${DIR_LIB_SRCS}")
    set(CMAKE_CXX_FLAGS "-DPROJECT_BASE_${module_name}=1 ${CMAKE_CXX_FLAGS}")

    include(obj_name_provide)
    set(module ${module_name})
    set(method_module_name ${module} CACHE INTERNAL "module name")
    obj_name_provide(CMAKE_BUILD_TYPE project_base_build_shared_lib module)

     #生成链接库
    add_library(${${module}_target_name} ${${module}_build_link_type} ${DIR_LIB_SRCS})

    ## install method 
    message(STATUS ${CMAKE_CURRENT_LIST_FILE})
    file(
        GLOB
        ${module}_files
        *.hpp
    )
    install(FILES ${${module}_head} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/${module})

    include(test_engineer_config)
    build_test_option(${module})
    if(${_build_${module}_test})
        add_subdirectory(${module}_test)
    else()
    endif()
endmacro(add_base_module)