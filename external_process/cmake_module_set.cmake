# this script add the dir of cmake_module to the CMAKE_MODULE_PATH
message(STATUS ${CMAKE_CURRENT_LIST_FILE})
get_filename_component(ProjectBase_Root ${CMAKE_CURRENT_LIST_DIR} PATH)
set(CMAKE_MODULE_PATH ${ProjectBase_Root}/cmake_module/ ${CMAKE_MODULE_PATH})