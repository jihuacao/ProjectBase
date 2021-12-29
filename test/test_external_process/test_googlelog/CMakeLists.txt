cmake_minimum_required(VERSION 3.5)
project(test_googlelog)

get_filename_component(test_external_process_dir ${${PROJECT_NAME}_SOURCE_DIR} PATH)
get_filename_component(test_dir ${test_external_process_dir} PATH)
get_filename_component(src_dir ${test_dir} PATH)

set(CMAKE_MODULE_PATH ${src_dir}/external_process/ ${CMAKE_MODULE_PATH})

include(cmake_module_set)
include(common_project_configure)
project_common_setting()
include(external_setting)
external_cmake_args()
include(popular_message)
cmakelists_base_header()
project_base_system_message()
message(STATUS ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG})

include(../../../external_process/glog.cmake)
glog_target()
show_target_properties(${${glog_module_name}_target_name})
get_target_property(in ${${glog_module_name}_target_name} IMPORTED_SONAME_RELEASE)

set(source_file
    "${test_googlelog_SOURCE_DIR}/test_googlelog.cpp"
    )
add_executable(${PROJECT_NAME} ${source_file})
target_link_libraries(${PROJECT_NAME} ${${glog_module_name}_target_name})
add_dependencies(${PROJECT_NAME} ${${glog_module_name}_target_name})