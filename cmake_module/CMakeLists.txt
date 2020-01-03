## install cmake_module
include(popular_message)
cmakelists_base_header()
set(module cmake_module)
file(
    GLOB
    ${module}_files
    *.cmake
    )
install(FILES ${${module}_files} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/${module})
