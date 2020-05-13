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
#ifndef PROJECT_BASE_TENSOR_TENSOR_H
#define PROJECT_BASE_TENSOR_TENSOR_H
#include <ProjectBase/tensor/Define.hpp>
#include <boost/test/data/monomoriphic/initializer_list.hpp>

namespace ProjectBase{
    namespace Tensor{
        class PROJECT_BASE_TENSOR_SYMBOL Shape{
            public:
                Shape(const boost::monomorphic::init_list<long>& sl);
            public:
                long get(int index);
        };

        class PROJECT_BASE_TENSOR_SYMBOL Tensor{
            public:
                Tensor();
            public:
                void shape();
                void type();
            public:
                void T();
        };
    }
}
#endif