include(popular_message)
cmakelists_base_header()

include(obj_name_provide)
set(module DeviceInfo)
set(device_info_module_name ${module} CACHE INTERNAL "module name")
obj_name_provide(CMAKE_BUILD_TYPE project_base_build_shared_lib module)

project(${${module}_target_name} CXX)

 # generate executable
list(APPEND PRIVATE_C_COMPILE_DEFINE "")
list(APPEND PRIVATE_CXX_COMPILE_DEFINE "")

file(GLOB test_${module}_src *.h *.cc)
debug_the_src_with_abspath(test_${module}_src)
add_executable(${${device_info_module_name}_target_name} ${test_${module}_src})
# compile
target_compile_definitions(${${device_info_module_name}_target_name} PRIVATE ${PRIVATE_CXX_COMPILE_DEFINE})
# link
target_link_libraries(${${device_info_module_name}_target_name} ${${opencl_base_module_name}_target_name})
target_link_libraries(${${device_info_module_name}_target_name} ${${gtest_module_name}_target_name})


## obj
install(
    TARGETS
    ${${device_info_module_name}_target_name}
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin 
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)