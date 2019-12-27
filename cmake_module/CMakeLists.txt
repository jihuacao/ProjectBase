## install cmake_module
set(module cmake_module)
file(
    GLOB
    ${module}_files
    *.cmake
    )
install(FILES ${${module}_files} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/${module})
