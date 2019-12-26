aux_source_directory(. DIR_LIB_SRCS)
set(CMAKE_CXX_FLAGS "-DPROJECT_BASE_MULTI_LANGUAGE=1 ${CMAKE_CXX_FLAGS}")

include(obj_name_provide)
set(Obj multi_language)
obj_name_provide(CMAKE_BUILD_TYPE project_base_build_shared_lib Obj)

 #生成链接库
add_library(${${Obj}_target_name} ${${Obj}_build_link_type} ${DIR_LIB_SRCS})

# install multilanguage
## header
set(
    multi_language_files
    "multi_language.hpp"
    "Define.hpp"
    )
install(FILES ${multi_language_files} DESTINATION  ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/multi_language)

## obj
message("${CMAKE_CURRENT_LIST_DIR}")
install(
    TARGETS
    ${${Obj}_target_name}
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/${PROJECT_NAME} 
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME})