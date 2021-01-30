#include <ProjectBase/util/sort.hpp>
#include <gtest/gtest.h>
#include <iostream>


TEST(Sort, QuickSort){
    int len = 10;
    int a[len] = {0, 1, 2, 2, 3, 4, 5, 6, 7, 8};
    quick_sort(a, 0, len);
    for(auto i = 0; i < len; ++i) std::cout << a[i];
    std::cout << std::endl;
}