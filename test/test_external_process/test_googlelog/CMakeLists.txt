cmake_minimum_required(VERSION 3.5)
project(test_googlelog)

get_filename_component(test_external_process_dir ${${PROJECT_NAME}_SOURCE_DIR} PATH)
get_filename_component(test_dir ${test_external_process_dir} PATH)
get_filename_component(src_dir ${test_dir} PATH)

set(CMAKE_MODULE_PATH ${src_dir}/external_process/ ${CMAKE_MODULE_PATH})
include(cmake_module_set)
include(external_setting)
include(glog)
#display_properties_supported()
#show_target_properties(${glog_target_name})

set(source_file
    "${test_googlelog_SOURCE_DIR}/test_googlelog.cpp"
    )
add_executable(${PROJECT_NAME} ${source_file})
target_link_libraries(${PROJECT_NAME} ${glog_target_name})
add_dependencies(${PROJECT_NAME} ${glog_target_name})

#
#add_library(test SHARED ${source_file})
#target_include_directories(test PRIVATE ${external_install_path}/include)
#target_link_directories(test PRIVATE ${external_install_path}/lib)
#target_link_libraries(test PUBLIC libglog.so)
#add_dependencies(test ${PROJECT_NAME})
#
#add_executable(test2 "${test_googlelog_SOURCE_DIR}/test_dll.cpp")
#target_link_libraries(test2 PUBLIC test)
#add_dependencies(test2 test)