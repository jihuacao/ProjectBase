#include <vector>
#include <gtest/gtest.h>
#include "ProjectBase/basic_algorithm/binary_tree.hpp"


ProjectBase::Tree::BinaryTree<ProjectBase::Tree::binary_tree_node<int>> make_binary_tree(){
    auto tree = ProjectBase::Tree::BinaryTree<ProjectBase::Tree::binary_tree_node<int>>();
    return tree;
}


TEST(TestBinaryTree, TestBalanced){
    std::vector<int> temp = {1, 2, 3, 4, 5, 5};
    int i = 0;
    auto b = i < 3;
    auto a = i < temp.size();
    for(; i < 3; ++i){
        temp.pop_back();
    }
    std::cout << i << std::endl;
    std::cout << i << std::endl;
}