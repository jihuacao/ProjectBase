include(popular_message)
cmakelists_base_header()
aux_source_directory(. DIR_LIB_SRCS)
set(CMAKE_CXX_FLAGS "-DPROJECT_BASE_UI_STYLE_CONFIG=1 ${CMAKE_CXX_FLAGS}")

include(obj_name_provide)
set(Obj UiStyleConfig)
obj_name_provide(CMAKE_BUILD_TYPE project_base_build_shared_lib Obj)

 #生成链接库
add_library(${${Obj}_target_name} ${${Obj}_build_link_type} ${DIR_LIB_SRCS})
target_include_directories(${${Obj}_target_name} PUBLIC ${Boost_INCLUDE_DIRS})
target_link_libraries(${${Obj}_target_name} PUBLIC ${Boost_LIBRARIES})

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
    ${${Obj}_target_name}
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin 
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
