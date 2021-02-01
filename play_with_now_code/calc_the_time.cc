#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <vector>

TEST(CalcTheTime, Int){
    int parrellel = 4;
    int time = 5;
    std::vector<int> task = {5, 4, 1, 1, 1};

    int remain = 0;
    for(auto i = 0; i < time; ++i){
        remain = (task[i] + remain) <= parrellel ? 0 : task[i] + remain - parrellel;
    }
    auto more_time = remain == 0 ? 0 : remain <= parrellel ? 1 : (remain / parrellel + (remain % parrellel == 0 ? 0 : 1));

    std::cout << time + more_time << std::endl;
}