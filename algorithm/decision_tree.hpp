/****************************************************************************
<one line to give the program's name and a brief idea of what it does.>
Copyright (C) <2021>  <CJH>
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

namespace ProjectBase{
    namespace algorithm{
        class Data{

        };
        class DecisionTree{
            public:
                virtual void train(const Data& data)=0;
                virtual void predict()=0;
        };
        class IncrementalDecisionTree: DecisionTree{
            public:
                virtual void continue_train(const Data& data);
        };
    };
}