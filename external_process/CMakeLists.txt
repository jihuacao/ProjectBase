## install external_process
message(STATUS ${CMAKE_CURRENT_LIST_FILE})
set(module external_process)
file(
    GLOB
    ${module}_files
    *.cmake
    )
install(FILES ${${module}_files} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/${module})
