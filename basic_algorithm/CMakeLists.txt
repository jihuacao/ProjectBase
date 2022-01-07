## compile basic_algorithm 
include(popular_message)
cmakelists_base_header()
aux_source_directory(. DIR_LIB_SRCS)
list(APPEND PRIVATE_COMPILE_DEFINE "PROJECT_BASE_BASIC_ALGORITHM=1")

include(obj_name_provide)
set(module BasicAlgorithm)
set(basic_algorithm_module_name ${module} CACHE INTERNAL "module name")
obj_name_provide(CMAKE_BUILD_TYPE project_base_build_shared_lib module)

 #生成链接库
add_library(${${module}_target_name} ${${module}_build_link_type} ${DIR_LIB_SRCS})
target_compile_definitions(${${module}_target_name} PRIVATE ${PRIVATE_COMPILE_DEFINE})

## install basic_algorithm 
message(STATUS ${CMAKE_CURRENT_LIST_FILE})
set(module basic_algorithm)
file(
    GLOB
    ${module}_files
    *.hpp
)
install(FILES ${${module}_head} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/${module})

include(test_engineer_config)
build_test_option(${module})
if(${_build_${module}_test})
    add_subdirectory(basic_algorithm_test)
else()
endif()

