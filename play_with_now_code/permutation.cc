#include <gtest/gtest.h>
#include <gtest/gtest-param-test.h>
#include <iostream>
#include <random>
#include <vector>
#include <unordered_map>
#include <queue>

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