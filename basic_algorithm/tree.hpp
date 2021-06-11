/****************************************************************************
<one line to give the program's name and a brief idea of what it does.>
Copyright (C) <2021>  <JiHuaCao>
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
#include <ProjectBase/basic_algorithm/Define.hpp>

namespace ProjectBase{
    namespace Tree{
        template<typename NodeType, typename TreeSizeType, typename DepthSizeType> 
        class Tree{
            public:
            virtual bool empty() const=0;
            virtual const NodeType& root() const=0;
            virtual DepthSizeType depth() const=0;
            virtual NodeType& value(TreeSizeType index) const=0;
            virtual NodeType& assign(const NodeType& source, TreeSizeType index)=0;
            virtual NodeType& parent(TreeSizeType index) const=0;
            virtual NodeType& child(TreeSizeType parent_index, TreeSizeType child_index) const=0;
            virtual TreeSizeType child_size(TreeSizeType parent_index) const=0;
            virtual NodeType&& delete_child(TreeSizeType parent_index, )
        };

        //class chain_tree;
        //class ChainTree: Tree<NodeType>{
        //    public:
        //    ChainTree();
        //    chain_tree* _impl;
        //};
    };
};