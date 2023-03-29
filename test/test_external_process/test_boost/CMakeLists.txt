cmake_minimum_required(VERSION 3.5)
project(test_boost)

get_filename_component(test_external_process_dir ${${PROJECT_NAME}_SOURCE_DIR} PATH)
get_filename_component(test_dir ${test_external_process_dir} PATH)
get_filename_component(src_dir ${test_dir} PATH)

set(CMAKE_MODULE_PATH ${src_dir}/external_process/ ${CMAKE_MODULE_PATH})
include(cmake_module_set)
include(ProjectBaseSet)
include(external_setting)
external_cmake_args()
include(boost)
#show_target_properties(${boost_target_name})

set(sources "${CMAKE_CURRENT_SOURCE_DIR}/test_boost.cpp")
add_executable(${PROJECT_NAME} ${sources})
target_compile_options(${PROJECT_NAME} PRIVATE -std=c++11)
target_link_libraries(${PROJECT_NAME} ${boost_target_name}) 
add_dependencies(${PROJECT_NAME} ${boost_target_name})
#target_link_libraries(${PROJECT_NAME} Boost::boost) 
#add_dependencies(${PROJECT_NAME} Boost::boost)
show_target_properties(${PROJECT_NAME})

add_executable(
    app 
    "${CMAKE_CURRENT_SOURCE_DIR}/app.cpp"
    )
target_compile_options(app PRIVATE -std=c++11)
target_link_libraries(app ${boost_target_name} pthread) 
add_dependencies(app ${boost_target_name})
show_target_properties(app)