message(STATUS ${CMAKE_CURRENT_LIST_FILE})
aux_source_directory(. DIR_SRCS)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

include(obj_name_provide)
set(Obj TestUiStyleConfig)
executable_name_provide(CMAKE_BUILD_TYPE Obj)