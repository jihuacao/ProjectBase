include(popular_message)
cmakelists_base_header()

include(obj_name_provide)
set(module PluginManagerTest)
obj_name_provide(CMAKE_BUILD_TYPE project_base_build_shared_lib module)

 # generate executable
set(CMAKE_CXX_FLAGS "-DPROJECT_BASE_FUNCTION=1 ${CMAKE_CXX_FLAGS}")

file(GLOB test_${module}_src *.h *.cc)
file(GLOB test_${module}_src_debug RELATIVE ${CMAKE_CURRENT_LIST_DIR} *.h *.cc)
message(DEBUG "the src for the test_${module}: ${test_${module}_src_debug}")
add_executable(${${module}_target_name} ${test_${module}_src})
target_link_libraries(${${module}_target_name} ${${plugin_manager_module_name}_target_name}) 
target_link_libraries(${${module}_target_name} ${glog_target_name})
target_link_libraries(${${module}_target_name} ${gflags_target_name})

# install multilanguage
## obj
install(
    TARGETS
    ${${module}_target_name}
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin 
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)