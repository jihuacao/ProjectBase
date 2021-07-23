include(popular_message)
cmakelists_base_header()

include(obj_name_provide)
set(module ProgramOpencl)
set(program_opencl_module_name ${module} CACHE INTERNAL "module name")
obj_name_provide(CMAKE_BUILD_TYPE project_base_build_shared_lib module)

 # generate executable
set(CMAKE_CXX_FLAGS "-DPROJECT_BASE_FUNCTION=1 ${CMAKE_CXX_FLAGS} -fpermissive")

file(GLOB test_${module}_src *.h *.cc)
debug_the_src_with_abspath(test_${module}_src)
add_executable(${${program_opencl_module_name}_target_name} ${test_${module}_src})
target_link_libraries(${${program_opencl_module_name}_target_name} ${${gtest_module_name}_target_name})
target_link_libraries(${${program_opencl_module_name}_target_name} ${${gflags_module_name}_target_name})


## obj
install(
    TARGETS
    ${${program_opencl_module_name}_target_name}
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin 
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
