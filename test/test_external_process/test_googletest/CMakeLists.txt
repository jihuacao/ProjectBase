cmake_minimum_required(VERSION 3.5)

project(test_googletest)

get_filename_component(test_external_process_dir ${test_googletest_SOURCE_DIR} PATH)
get_filename_component(test_dir ${test_external_process_dir} PATH)
get_filename_component(src_dir ${test_dir} PATH)

set(CMAKE_MODULE_PATH ${src_dir}/external_process/ ${CMAKE_MODULE_PATH})

include(cmake_module_set)
include(external_setting)
include(popular_message)
cmakelists_base_header()
project_base_system_message()

include(GTest)
show_target_properties(${GTest_target_name})
get_target_property(in ${GTest_target_name} IMPORTED_SONAME_RELEASE)

set(sources "${${PROJECT_NAME}_SOURCE_DIR}/test_googletest.cpp")
add_executable(${PROJECT_NAME} ${sources})
target_link_libraries(${PROJECT_NAME} ${GTest_target_name})
target_compile_options(${PROJECT_NAME} PRIVATE -std=c++11)
add_dependencies(${PROJECT_NAME} ${GTest_target_name})