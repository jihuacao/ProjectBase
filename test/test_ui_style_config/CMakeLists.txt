message(STATUS ${CMAKE_CURRENT_LIST_FILE})
aux_source_directory(. DIR_SRCS)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

include(obj_name_provide)
set(Obj TestUiStyleConfig)
executable_name_provide(CMAKE_BUILD_TYPE Obj)

##生成链接库
#project(${${Obj}_target_name})
#add_executable(${${Obj}_target_name})
#target_sources(${${Obj}_target_name} PRIVATE ${DIR_SRCS})
#target_compile_definitions(${${Obj}_target_name} "")
#target_compile_options(${${Obj}_target_name} "")
#target_compile_features(${${Obj}_target_name})
#target_include_directories(${${Obj}_target_name} ${Boost_INCLUDE_DIRS})
#target_link_libraries(${${Obj}_target_name} ${Boost_LIBRARIES})
#add_definitions(${${Obj}_target_name})
#
## install executable
#install(
#    TARGETS
#    ${${Obj}_target_name}
#    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin 
#    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
#    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

