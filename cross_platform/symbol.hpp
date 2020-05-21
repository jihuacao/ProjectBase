#ifndef PROJECT_BASE_CROSS_PLATFORM_H
#define PROJECT_BASE_CROSS_PLATFORM_H
#include <ProjectBase/cross_platform/util_method.hpp>
// cross platform symbol export or import
#if defined(_WIN32) || defined(__CYGWIN__) || defined(WIN32) || defined(WIN64)

#ifdef __GNUC__ // cygwin
#define SYMBOL_EXPORT __attribute__((dllexport))
#define SYMBOL_IMPORT __attribute__((dllimport))
#define SYMBOL_LOCAL
#else // windows, msvc
#define SYMBOL_EXPORT __declspec(dllexport)
#define SYMBOL_IMPORT __declspec(dllimport)
#define SYMBOL_LOCAL
#endif

#else
#if __GNUC__ >= 4
#define SYMBOL_EXPORT __attribute__((visibility("default")))
#define SYMBOL_IMPORT
#define SYMBOL_LOCAL __attribute__((visibility("hidden")))
#else
#define SYMBOL_EXPORT
#define SYMBOL_LOCAL
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