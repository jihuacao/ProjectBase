/****************************************************************************
<one line to give the program's name and a brief idea of what it does.>
Copyright (C) <year>  <name of author>
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
/**
 * @file 文件名
 * @brief 简介
 * @details 细节
 * @mainpage 工程概览
 * @author 作者
 * @email 邮箱
 * @version 版本号
 * @date 年-月-日
 * @license 版权
 */
#include <gtest/gtest.h>
#include <gtest/gtest-param-test.h>
#include <ProjectBase/method/char_operation.hpp>

TEST(A, B){
    char* a = "12345";
    char* b = "12345";
    char* result_ptr = nullptr;
    size_t result_size = 0;
    ProjectBase::Method::CharOperation::Add<u_char>(a, 5, b, 5, &result_ptr, &result_size);
    std::cout << std::string(result_ptr) << std::endl;
    ASSERT_TRUE(std::string(result_ptr) == "24690");
}