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
#include <sys/types.h>
#include <ProjectBase/method/Define.hpp>
/**
 * \brief CharOperation
 * \note note 实现基于字符串的数学运算
 * \author JiHuaCao
 * \since 0.1
 * */
namespace ProjectBase{
    namespace Method{
        class CharOperation {
            public:
                CharOperation(){};
            public:
                /**
                 * \brief 大数相加
                 * \note note
                 * \author JiHuaCao
                 * \param[in] char* a_ptr
                 * \param[in] size_t a_size
                 * \param[in] char* b_ptr
                 * \param[in] size_t b_size
                 * \param[inout] char** result_ptr
                 * \param[inout] size* result_size
                 * \return return unsigned long long
                 * \retval retval
                 *      0： 成功
                 *      1： 不规范输入
                 * \since version 0.1
                 * */
                template<typename DT> static unsigned long long  Add(
                    char* a_ptr, size_t a_size, char* b_ptr, size_t b_size, char** result_ptr, size_t* result_size){
                        *result_size = std::max(a_size, b_size) + 1;
                        *result_ptr = malloc(*result_size * sizeof(DT));
                        DT temp = 0;
                        DT up = 0;
                        std::vector<DT> _result;
                        for(int i = 0, j = 0; i < a_size && j < b_size; ++i, ++j){
                            temp = *((DT*)a_ptr + a_size - i - 1) + *((DT*)b_ptr + b_size - i -1) + up;

                            if(temp >= 10){
                                _result.push_back(temp - 10);
                                //*((DT*)(*result_ptr) + *result_size - i - 1) = temp - 10;
                                up = 1;
                            }
                            else{
                                _result.push_back(temp);
                                //*((DT*)(*result_ptr) + *result_size - i - 1) = temp;
                                up = 0;
                            }
                        }
                        if (a_size > b_size){
                            while(a_size - _result.size() - 1 != 0){
                                _result.push_back(*((DT*)a_ptr + a_size - _result.size() - 1) + up);
                            }
                        }
                        else if (a_size < b_size){
                            while(b_size - _result.size() - 1 != 0){
                                _result.push_back(*((DT*)a_ptr + a_size - _result.size() - 1) + up);
                            }
                        }
                        else{
                            if (up == 1){
                                _result.push_back(1);
                            }
                        }
                        return 0;
                        };
        };
    };
};