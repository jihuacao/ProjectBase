aux_source_directory(. DIR_LIB_SRCS)
set(CMAKE_CXX_FLAGS "-DPROJECT_BASE_MULTI_LANGUAGE=1 ${CMAKE_CXX_FLAGS}")
# 生成链接库
add_library(MultiLanguage SHARED ${DIR_LIB_SRCS})