#include <gtest/gtest.h>
#include <gtest/gtest-param-test.h>
#include <iostream>
#include <random>
#include <vector>

struct total_input{
    void* ptr;
    size_t size;
    public:
    total_input(void* ptr, size_t size){ptr = ptr; size = size;};
};

std::vector<total_input> generate_data(){
    return std::vector<total_input> ();
}

class Total:
public testing::TestWithParam<total_input>{

};

TEST_P(Total, Permutation){

};

TEST_P(Total, K){

};

//INSTANTIATE_TEST_SUITE_P(Permutaion, Total, testing::Values(total_input(nullptr, 1), total_input(nullptr, 1)));
INSTANTIATE_TEST_SUITE_P(Permutaion, Total, testing::Values(generate_data()));
