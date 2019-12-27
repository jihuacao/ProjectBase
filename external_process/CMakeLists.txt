## install external_process
set(module external_process)
file(
    GLOB
    ${module}_files
    *.cmake
    )
install(FILES ${${module}_files} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/${module})
