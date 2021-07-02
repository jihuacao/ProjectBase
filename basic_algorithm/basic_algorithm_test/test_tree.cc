#include <gtest/gtest.h>
#include <iostream>

#include "ProjectBase/basic_algorithm/tree.hpp"

ProjectBase::Tree::tree<ProjectBase::Tree::duplex_tree_node<int>> make_tree(){
    auto tree = ProjectBase::Tree::tree<ProjectBase::Tree::duplex_tree_node<int>>(0);
    auto iter = tree.begin();
    tree.append_child(iter, 1);
    tree.append_child(iter, 2);
    tree.append_child(iter, 3);
    tree.append_child(iter, 4);

    ++iter;

    ++iter;
    tree.append_child(iter, 5);

    ++iter;
    ++iter;
    tree.append_child(iter, 6);
    tree.append_child(iter, 7);

    ++iter;
    ++iter;
    ++iter;
    tree.append_child(iter, 8);
    tree.append_child(iter, 9);

    ++iter;
    tree.append_child(iter, 10);

    return tree;
}

TEST(TestTree, PreOrder){
    auto tree = make_tree();
    std::cout << "pre order" << std::endl;
    for(auto iter_pre = tree.begin(); iter_pre != tree.end(); ++iter_pre){
        std::cout << iter_pre.node->data << "->";
    }
    std::cout << std::endl;
}

TEST(TestTree, PostOrder){
    auto tree = make_tree();
    std::cout << "post order" << std::endl;
    for(auto iter_post = tree.begin_post(); iter_post != tree.end_post(); ++iter_post){
        std::cout << iter_post.node->data << "->";
    }
    std::cout << std::endl;
}