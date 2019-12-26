
## install cross_platform
set(module cross_platform)
set(
    cross_platform_head
    "symbol.hpp"
    "util_method.hpp"
)
install(FILES ${cross_platform_head} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/cross_platform)
