## install external_process
set(
    external_process_files
    "cmake_module_set.cmake"
    "boost.cmake"
    "external_download.cmake"
    "external_setting.cmake"
    "googleflags.cmake"
    "googlelog.cmake"
    "googletest.cmake"
    )
install(FILES ${external_process_files} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/external_process)
