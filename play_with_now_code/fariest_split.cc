#include <gtest/gtest.h>
#include <vector>
#include <iostream>

TEST(FariestSplit, Int){
    std::vector<int> pool = {2, 1, 3, 4, 6, 5, 7, 1000, 100, 100};
    int temp = 0;
    for(auto j =1; j < pool.size(); ++j){
        for (auto i = 1; i < pool.size(); ++i){
            if(pool[i] > pool[i - 1]){
                temp = pool[i - 1];
                pool[i - 1] = pool[i];
                pool[i] = temp;
            }
        }
    }
    int a_sum = 0, a_amount = 0, b_sum = 0, b_amount = 0;
    std::vector<int> a_content;
    std::vector<int> b_content;
    for(auto p = pool.begin(); p != pool.end(); ++p){
        if(a_amount == 5){
            b_amount += 1;
            b_sum += *p;
            b_content.push_back(*p);
            continue;
        }
        else if(b_amount == 5){
            a_amount += 1;
            a_sum += *p;
            a_content.push_back(*p);
            continue;
        }
        if(a_sum == b_sum){
            if(a_amount <= b_amount){
                a_amount += 1;
                a_sum += *p;
                a_content.push_back(*p);
            }
            else{
                b_amount += 1;
                b_sum += *p;
                b_content.push_back(*p);
            }
            continue;
        }
        if(a_sum < b_sum){
            a_sum += *p;
            a_amount += 1; 
            a_content.push_back(*p);
            continue;
        }
        if(a_sum > b_sum){
            b_sum += *p;
            b_amount += 1;
            b_content.push_back(*p);
            continue;
        }
    }
    int sub = a_sum >= b_sum ? a_sum - b_sum : b_sum - a_sum;
    std::cout << sub << std::endl;
}
