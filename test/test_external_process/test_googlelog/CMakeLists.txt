cmake_minimum_required(VERSION 3.5)
project(test_googlelog)

get_filename_component(test_external_process_dir ${${PROJECT_NAME}_SOURCE_DIR} PATH)
get_filename_component(test_dir ${test_external_process_dir} PATH)
get_filename_component(src_dir ${test_dir} PATH)

set(CMAKE_MODULE_PATH ${src_dir}/external_process/ ${CMAKE_MODULE_PATH})
include(cmake_module_set)
include(external_setting)
include(googlelog)

set(source_file
    "${test_googlelog_SOURCE_DIR}/test_googlelog.cpp"
    )
message(STATUS ${PROJECT_NAME})
add_executable(${PROJECT_NAME} ${source_file})
add_definitions(${PROJECT_NAME} glog)