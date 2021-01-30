/****************************************************************************
<tensor.cpp>
Copyright (C) <2020/5/12>  <JiHua Cao>
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
#include <iostream>
#include <glog/logging.h>
#include "ProjectBase/tensor/inner_tensor.h"


ProjectBase::Tensor::Tensor::Tensor()
    :_impl(new ProjectBase::Tensor::inner_tensor())
{
}

ProjectBase::Tensor::Tensor::Tensor(const ProjectBase::Tensor::Tensor& other)
    :_impl(new ProjectBase::Tensor::inner_tensor(*other._impl))
{
}

ProjectBase::Tensor::Tensor::Tensor(ProjectBase::Tensor::Tensor&& ref)
    :_impl(new ProjectBase::Tensor::inner_tensor(std::move(*ref._impl)))
{

}

ProjectBase::Tensor::Tensor::~Tensor()
{
    delete _impl;
    _impl = nullptr;
}