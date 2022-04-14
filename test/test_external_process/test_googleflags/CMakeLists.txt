cmake_minimum_required(VERSION 3.5)
project(test_googleflags)

get_filename_component(test_external_process_dir ${test_googleflags_SOURCE_DIR} PATH)
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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -save-temps")

include(../../../external_process/gflags.cmake)
gflags_target()

set(sources "${${PROJECT_NAME}_SOURCE_DIR}/test_gflags.cpp")
add_executable(${PROJECT_NAME} ${sources})
target_link_libraries(${PROJECT_NAME} ${gflags_target_name})
add_dependencies(${PROJECT_NAME} ${gflags_target_name})