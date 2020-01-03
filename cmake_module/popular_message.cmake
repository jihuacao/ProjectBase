macro(project_base_system_message)
message("your system name is ${CMAKE_SYSTEM_NAME}, using CMAKE_SYSTEM_NAME to get the system name--${CMAKE_CURRENT_LIST_FILE}")
endmacro(project_base_system_message)

macro(cmakelists_base_header)
message(STATUS ${CMAKE_CURRENT_LIST_FILE})
endmacro(cmakelists_base_header)