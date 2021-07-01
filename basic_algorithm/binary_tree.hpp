#ifndef PROJECT_BASE_ALGORITHM_BINARY_TREE_H
#define PROJECT_BASE_ALGORITHM_BINARY_TREE_H
#include <ProjectBase/basic_algorithm/tree.hpp>
namespace ProjectBase{
    namespace Tree{
        template<typename T>
        class _binary_tree_node: _tree_node<T>{

        };

        template<typename Node>
        class _binary_tree: _tree<Node::T>{

        };

        template<class T>
        class binary_tree_node: _binary_tree_node<T>{
            public:
                binary_tree_node<T>* left() const;
                binary_tree_node<T>* right() const;
            public:
                binary_tree_node<T>* _left_node;
                binary_tree_node<T>* _right_node;
        };

        template<typename Node>
        class BinaryTree: _binary_tree<Node> {
            public:
                BinaryTree();
        };
    };
};
#endif