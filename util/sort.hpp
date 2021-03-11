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
template<typename _T> size_t _partition (_T* const L, size_t low, size_t high)
{
	int temp = L[low];
	int pt   = L[low]; //哨兵
	while (low != high)
	{
		while (low < high && L[high] >= pt)
			high--;
		L[low] = L[high];		
 
		while (low < high && L[low] <= pt)
			low++;
		L[high] = L[low];
	}	
	L[low] = temp;
	return low;
};

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
template<typename _T> void quick_sort(_T* const l, size_t low, size_t high){
    size_t pl;
    if (low < high) {
        pl = _partition(l, low, high);
        quick_sort(l, low, pl - 1);
        quick_sort(l, pl - 1, high);
    }
};