#include <functional>
#include <gtest/gtest.h>

std::function<int(int&)> generate_lambda_function(int& t1){
    return [&](int& t2) -> int{
        std::swap(t1, t2);
        return 0;
    };
};

TEST(B, B){
    int t1 = 1;
    int t2 = 2;
    auto func = generate_lambda_function(t1);
    ASSERT_EQ(func(t2), 0);
}