/****************************************************************************
<one line to give the program's name and a brief idea of what it does.>
Copyright (C) <2020>  <JiHua Cao>
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

*****************************************************************************/
#ifndef PROJECT_BASE_NUMBER_THEORY_DEFINE_H
#define PROJECT_BASE_NUMBER_THEORY_DEFINE_H
#include <ProjectBase/cross_platform/cross_platform.hpp>
#if defined(PROJECT_BASE_NUMBER_THEORY)
#define PROJECT_BASE_NUMBER_THEORY_SYMBOL SYMBOL_EXPORT
#else
#define PROJECT_BASE_NUMBER_THEORY_SYMBOL SYMBOL_IMPORT
#endif // !1PROJECT_BASE_NUMBER_THEORY

#pragma message(MACRO_TO_STRING(PROJECT_BASE_NUMBER_THEORY_SYMBOL))

#else
#endif // !1PROJET_BASE_NUMBER_THEORY_DEFINE_H



