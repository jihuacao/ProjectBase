/****************************************************************************
<codec.hpp>
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
#ifndef PROJECT_BASE_CODEC_CODEC_H
#define PROJECT_BASE_CODEC_CODEC_H
#include <ProjectBase/codec/Define.hpp>
#include <ProjectBase/tensor/tensor.hpp>
namespace ProjectBase{
    namespace Codec{
        class PROJECT_CODEC_SYMBOL Codec{
            public:
                Codec();

            public:
                ProjectBase::Tensor::Tensor encode(const ProjectBase::Tensor::Tensor& data) const;
                ProjectBase::Tensor::Tensor decode(const ProjectBase::Tensor::Tensor& bytes) const;
            public:
                void t(ProjectBase::Tensor::Tensor& data);
        };
    };
}
#else
#endif
