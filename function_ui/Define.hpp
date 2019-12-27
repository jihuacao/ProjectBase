#ifndef PROJECT_BASE_FUNCTION_UI_H
#define PROJECT_BASE_FUNCTION_UI_H
#include <ProjectBase/cross_platform/symbol.hpp>
#include <ProjectBase/cross_platform/util_method.hpp>
#ifndef PROJECT_BASE_FUNCTION_UI
#define PROJECT_BASE_FUNCTION_UI_SYMBOL SYMBOL_IMPORT
#else
#define PROJECT_BASE_FUNCTION_UI_SYMBOL SYMBOL_EXPORT
#endif // ! PROJECT_BASE_FUNCTION_UI
#pragma message(MACRO_TO_STRING(PROJECT_BASE_FUNCTION_UI_SYMBOL))
#endif // ! PROJECT_BASE_FUNCTION_UI_H