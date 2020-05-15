include(popular_message)


file(GLOB first LIST_DIRECTORIES OFF *)
message(STATUS "test debug the src with abspath first: ${first}")
debug_the_src_with_abspath(first)
#set(cache_for_CMAKE_CURRENT_LIST_DIR ${CMAKE_CURRENT_LIST_DIR})
#set(CMAKE_CURRENT_LIST_DIR ${CMAKE_CURRENT_LIST_DIR}/debug_the_src_with_abspath)
#file(GLOB first LIST_DIRECTORIES OFF *)
#message(STATUS "test debug the src with abspath second: ${first}")
#debug_the_src_with_abspath(first)
#set(CMAKE_CURRENT_LIST_DIR ${cache_for_CMAKE_CURRENT_LIST_DIR})
list(APPEND t a/b/c a/c/b a/b/a a/c/t)
debug_the_src_with_abspath(t)