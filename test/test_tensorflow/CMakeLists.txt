message(STATUS ${CMAKE_CURRENT_LIST_FILE})
aux_source_directory(. DIR_SRCS)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

include(obj_name_provide)
set(module TestTensorflow)
set(test_tensorflow_module_name ${module} CACHE INTERNAL "module name")
obj_name_provide(CMAKE_BUILD_TYPE project_base_build_shared_lib module)

# generate executable
set(CMAKE_CXX_FLAGS "-DPROJECT_BASE_FUNCTION=1 ${CMAKE_CXX_FLAGS}")

file(GLOB test_${module}_src *.h *.cc)
debug_the_src_with_abspath(test_${module}_src)
include_directories(/data2/process_data/caojihua/Download/tensorflow/tensorflow)
link_directories(/data2/process_data/caojihua/Download/tensorflow/bazel-bin/tensorflow)
add_executable(${${test_tensorflow_module_name}_target_name} ${test_${module}_src})
target_link_libraries(${${test_tensorflow_module_name}_target_name} ${${gtest_module_name}_target_name})
target_link_libraries(${${test_tensorflow_module_name}_target_name} libtensorflow_cc.so)


## obj
install(
    TARGETS
    ${${test_tensorflow_module_name}_target_name}
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin 
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)