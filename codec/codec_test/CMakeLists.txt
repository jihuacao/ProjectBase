include(popular_message)
cmakelists_base_header()

include(obj_name_provide)
set(module CodecTest)
obj_name_provide(CMAKE_BUILD_TYPE project_base_build_shared_lib module)

 # generate executable
set(CMAKE_CXX_FLAGS "-DPROJECT_BASE_FUNCTION=1 ${CMAKE_CXX_FLAGS}")

file(GLOB test_${module}_src *.h *.cc)
file(GLOB test_${module}_src_debug RELATIVE ${CMAKE_CURRENT_LIST_DIR} *.h *.cc)
message(DEBUG "the src for the test_${module}: ${test_${module}_src_debug}")
add_executable(${module} ${test_${module}_src})
target_link_libraries(${module} ${${tensor_module_name}_target_name})

# install multilanguage
## header
file(GLOB ${module}_install_heads *.hpp)
message(DEBUG "the install header of the lib: ${${module}_install_heads}")
install(FILES ${${module}_install_heads} DESTINATION  ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/${module})

## obj
install(
    TARGETS
    ${${module}_target_name}
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin 
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

