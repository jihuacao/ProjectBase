## install cross_platform
message(STATUS ${CMAKE_CURRENT_LIST_FILE})
set(module cross_platform)
file(
    GLOB
    ${module}_files
    *.hpp
)
add_library(cross_platform INTERFACE)
install(FILES ${${module}_head} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/${module})
