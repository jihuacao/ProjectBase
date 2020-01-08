cmake_minimum_required(VERSION 3.5)
project(test_googlelog)

get_filename_component(test_external_process_dir ${${PROJECT_NAME}_SOURCE_DIR} PATH)
get_filename_component(test_dir ${test_external_process_dir} PATH)
get_filename_component(src_dir ${test_dir} PATH)

set(CMAKE_MODULE_PATH ${src_dir}/external_process/ ${CMAKE_MODULE_PATH})
include(cmake_module_set)
include(external_setting)
include(glog)

set(source_file
    "${test_googlelog_SOURCE_DIR}/test_googlelog.cpp"
    )
add_executable(${PROJECT_NAME} ${source_file})
#target_include_directories(${PROJECT_NAME} PUBLIC ${external_install_path}/include)
target_link_directories(${PROJECT_NAME} PRIVATE ${external_install_path}/lib)
target_link_libraries(${PROJECT_NAME} ${external_install_path}/lib/libglog.so.0.4.0)
add_dependencies(${PROJECT_NAME} ${glog_target_name})

list(APPEND external_targets glog)
#external_configs_getter(external_targets PROJECT_NAME)