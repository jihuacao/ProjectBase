#include <gtest/gtest.h>

int main(int argc, char* argv[]){
    auto lambda = [](int a) { return a + 3; };
    lambda(4);
    testing::InitGoogleTest(&argc, argv);
    int a = RUN_ALL_TESTS();
    return a;
}