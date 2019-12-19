cmake_minimum_required(VERSION 3.5)
project(test_googletest)

get_filename_component(test_external_process_dir ${test_googletest_SOURCE_DIR} PATH)
get_filename_component(test_dir ${test_external_process_dir} PATH)
get_filename_component(src_dir ${test_dir} PATH)

set(CMAKE_MODULE_PATH ${src_dir}/external_process/)

include(googletest_setting)