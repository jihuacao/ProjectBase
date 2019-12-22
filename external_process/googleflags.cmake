message(${CMAKE_CURRENT_LIST_FILE})
include(popular_message)
include(ExternalProject)
project_base_system_message()
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    if(CMAKE_BUILD_TYPE MATCHES "Debug")
        #gflags_debug.lib
        #gflags_no_threads_debug.lib
    elseif(CMAKE_BUILD_TYPE MATCHES "Release")
    endif()
elseif(CMAKE_SYSTEM_NAME MATCHES "Linux")
    if(CMAKE_BUILD_TYPE MATCHES "Debug")
    elseif(CMAKE_BUILD_TYPE MATCHES "Release")
        #libgflags.so
        #libgflags_nothreas.so
    endif()
endif()