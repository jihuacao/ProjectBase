cmake_minimum_required(VERSION 3.5)

project(test_absl)

get_filename_component(test_external_process_dir ${${PROJECT_NAME}_SOURCE_DIR} PATH)
get_filename_component(test_dir ${test_external_process_dir} PATH)
get_filename_component(src_dir ${test_dir} PATH)

set(CMAKE_MODULE_PATH ${src_dir}/external_process/ ${CMAKE_MODULE_PATH})

include(cmake_module_set)
include(external_setting)
include(popular_message)
cmakelists_base_header()
project_base_system_message()

include(absl)

set(sources "${${PROJECT_NAME}_SOURCE_DIR}/test_absl.cpp")
add_executable(${PROJECT_NAME} ${sources})
target_link_libraries(${PROJECT_NAME} ${absl_target_name})
target_compile_options(${PROJECT_NAME} PRIVATE -std=c++11)
add_dependencies(${PROJECT_NAME} ${absl_target_name})

# install multilanguage
## header
file(
    GLOB
    ${Obj}_files
    *.hpp
    )
install(FILES ${${Obj}_files} DESTINATION  ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/${Obj})
## obj
install(
    TARGETS
    ${PROJECT_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin 
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)