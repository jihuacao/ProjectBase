macro(add_base_test target_module_name)
    include(popular_message)
    cmakelists_base_header()

    include(obj_name_provide)
    set(module ${target_module_name}Test)
    set(${module}_module_name ${module} CACHE INTERNAL "module name")
    obj_name_provide(CMAKE_BUILD_TYPE project_base_build_shared_lib module)

     # generate executable
    set(CMAKE_CXX_FLAGS "-DPROJECT_BASE_FUNCTION=1 ${CMAKE_CXX_FLAGS}")

    aux_source_directory(${CMAKE_CURRENT_LIST_DIR} ${module}_src)
    message(STATUS "${module} src: ${${module}_src}")
    debug_the_src_with_abspath(${module}_src)
    add_executable(${${${module}_module_name}_target_name} ${${module}_src})
    target_link_libraries(${${${module}_module_name}_target_name} ${${target_module_name}_target_name})
    target_link_libraries(${${${module}_module_name}_target_name} ${${gtest_module_name}_target_name})


    ## obj
    install(
        TARGETS
        ${${${module}_module_name}_target_name}
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin 
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endmacro(add_base_test)

macro(add_binary target_module_name)
    include(popular_message)
    cmakelists_base_header()

    include(obj_name_provide)
    set(module ${target_module_name}Test)
    set(${module}_module_name ${module} CACHE INTERNAL "module name")
    obj_name_provide(CMAKE_BUILD_TYPE project_base_build_shared_lib module)

     # generate executable
    set(CMAKE_CXX_FLAGS "-DPROJECT_BASE_FUNCTION=1 ${CMAKE_CXX_FLAGS}")

    aux_source_directory(${CMAKE_CURRENT_LIST_DIR} ${module}_src)
    message(STATUS "${module} src: ${${module}_src}")
    debug_the_src_with_abspath(${module}_src)
    add_executable(${${${module}_module_name}_target_name} ${${module}_src})
    target_link_libraries(${${${module}_module_name}_target_name} ${${target_module_name}_target_name})
    target_link_libraries(${${${module}_module_name}_target_name} ${${gtest_module_name}_target_name})

    ## obj
    install(
        TARGETS
        ${${${module}_module_name}_target_name}
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin 
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endmacro(add_base_test)