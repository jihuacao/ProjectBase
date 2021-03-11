/****************************************************************************
<one line to give the program's name and a brief idea of what it does.>
Copyright (C) 2021  JiHua Cao
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
#include <ProjectBase/util/sort.hpp>

/**
 * \brief brief
 * \note note
 * \author none
 * \param[in] in
 * \param[out] out
 * \return return
 * \retval retval
 * \since version
 * */
template<typename _T> size_t search_max_min(_T* data_ptr, size_t size, _T* min, _T* max){
    return 0;
}

/**
 * \brief brief
 * \note note
 * \author none
 * \param[in] in
 * \param[out] out
 * \return return
 * \retval retval
 * \since version
 * */
template<typename _T> size_t search_k_max(_T* data_ptr, size_t low, size_t high, size_t k, _T* value, size_t* index){
	int temp;
	temp = _partition(data_ptr, low, high);
	if(temp == k-1)
	{
        *value = data_ptr[temp];
	}
	else if(temp > k-1)
		return search_k_max(data_ptr, low, temp - 1, k, value, index);
	else
		return search_k_max(data_ptr, temp + 1, high, k - temp, value, index);
};