#ifndef PROJECT_BASE_DATA_DEFINE_H
#define PROJECT_BASE_DATA_DEFINE_H
#include <ProjectBase/cross_platform/util_method.hpp>
#include <ProjectBase/cross_platform/symbol.hpp>
#if defined(PROJECT_BASE_DATA)
#define PROJECT_BASE_DATA_SYMBOL SYMBOL_EXPORT
#else
#define PROJECT_BASE_DATA_SYMBOL SYMBOL_IMPORT
#endif // !1PROJECT_BASE_DATA

#pragma message(MACRO_TO_STRING(PROJECT_BASE_DATA_SYMBOL))

#else
#endif // !1PROJET_BASE_DATA_DEFINE_H