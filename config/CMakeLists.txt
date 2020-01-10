## install external_process
include(popular_message)
include(external_setting)
cmakelists_base_header()
project_base_system_message()

set(module config)
file(
    GLOB
    ${module}_files
    *
    !CMakeLists.txt
    )
install(FILES ${${module}_files} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/${module})
