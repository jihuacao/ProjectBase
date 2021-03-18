#ifndef PROJECT_BASE_CROSS_PLATFORM_H
#define PROJECT_BASE_CROSS_PLATFORM_H
#include <ProjectBase/cross_platform/util_method.hpp>
// cross platform symbol export or import
#if defined(_WIN32) || defined(__CYGWIN__) || defined(WIN32) || defined(WIN64)
#else
#if __GNUC__ >= 4
#else
#endif
#endif

#pragma message(MACRO_TO_STRING(SYMBOL_EXPORT))
#pragma message(MACRO_TO_STRING(SYMBOL_IMPORT))
#pragma message(MACRO_TO_STRING(SYMBOL_LOCAL))

// #define SYMBOL(LIB_NAME) \
// #if defined(LIB_NAME) \
// #define SYMBOL_##LIB_NAME SYMBOL_EXPORT \
// #else \
// #define SYMBOL_##LIB_NAME SYMBOL_EXPORT \
// #endif

#else
#endif
