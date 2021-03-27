## compile method 
include(popular_message)
cmakelists_base_header()
aux_source_directory(. DIR_LIB_SRCS)
set(CMAKE_CXX_FLAGS "-DPROJECT_BASE_TENSOR=1 ${CMAKE_CXX_FLAGS} -fpermissive")

include(obj_name_provide)
set(module Method)
set(method_module_name ${module} CACHE INTERNAL "module name")
obj_name_provide(CMAKE_BUILD_TYPE project_base_build_shared_lib module)

 #生成链接库
add_library(${${module}_target_name} ${${module}_build_link_type} ${DIR_LIB_SRCS})
#target_link_libraries(${${module}_target_name} PUBLIC ${boost_target_name})
#target_link_libraries(${${module}_target_name} PUBLIC ${eigen3_target_name})

## install method 
message(STATUS ${CMAKE_CURRENT_LIST_FILE})
set(module method)
file(
    GLOB
    ${module}_files
    *.hpp
)
install(FILES ${${module}_head} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/${module})

include(test_engineer_config)
build_test_option(${module})
if(${_build_${module}_test})
    add_subdirectory(method_test)
else()
endif()

