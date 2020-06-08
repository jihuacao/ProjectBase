/****************************************************************************
<one line to give the program's name and a brief idea of what it does.>
Copyright (C) <2020/5/21>  <JiHua Cao>
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
#ifndef PROJECT_BASE_TENSOR_TENSOR_SHAPE_H
#define PROJECT_BASE_TENSOR_TENSOR_SHAPE_H
#include <ProjectBase/tensor/Define.hpp>

namespace ProjectBase{
    namespace Tensor{
        /**
         * \brief brief
         * \note note
         * \author none
         * \since version
         * */
        class PROJECT_BASE_TENSOR_SYMBOL TensorShape{
            public:
                TensorShape();
            public:
                long get(int index);
        };
    }
}
#endif