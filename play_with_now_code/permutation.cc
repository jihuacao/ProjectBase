#include <gtest/gtest.h>
#include <gtest/gtest-param-test.h>
#include <iostream>
#include <random>
#include <vector>
#include <unordered_map>
#include <queue>

/**
 * \brief brief TotalPermutation.Unrepeated
 * \note note 题目要求：给定一个vector，里面不包含重复元素，程序给出所有可能排序的结果
 * \author none
 * \since version
 * */
typedef int TotalPermutationUnrepeatedInputType;
typedef std::vector<TotalPermutationUnrepeatedInputType> TotalPermutationUnrepeatedInput;
typedef std::unordered_map<TotalPermutationUnrepeatedInputType, int> TotalPermutationStatus;
typedef std::vector<TotalPermutationUnrepeatedInput> TotalPermutationResult;
bool search(TotalPermutationStatus& status, std::queue<TotalPermutationUnrepeatedInputType>& remain_queue, TotalPermutationResult& result){
    return false;
};
class TotalPermutation:
public testing::TestWithParam<TotalPermutationUnrepeatedInput>{
};
TEST_P(TotalPermutation, Unrepeated){
};
INSTANTIATE_TEST_SUITE_P(Permutaion, TotalPermutation, testing::Values(TotalPermutationUnrepeatedInput({1, 2, 3, 4})));