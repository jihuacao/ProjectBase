## install cmake_module
set(
    cmake_modules_files
    "popular_message.cmake"
    )
install(FILES ${cmake_module_files} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME})
