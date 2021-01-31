//#include <ProjectBase/util/sort.hpp>
//#include <gtest/gtest.h>
//
//int main(int argc, char** argv){
//    testing::InitGoogleTest(&argc, argv);
//    auto a = RUN_ALL_TESTS();
//    return a;
//}
/*
#include <iostream>
#include <random>
#include <vector>

int main(int argc, char**argv){
    int parrellel = 4;
    int time = 5;
    std::vector<int> task = {5, 4, 1, 1, 1};

    int remain = 0;
    for(auto i = 0; i < time; ++i){
        remain = (task[i] + remain) <= parrellel ? 0 : task[i] + remain - parrellel;
    }
    auto more_time = remain == 0 ? 0 : remain <= parrellel ? 1 : (remain / parrellel + (remain % parrellel == 0 ? 0 : 1));

    std::cout << time + more_time << std::endl;
    return 0j
}*/

//#include <string>
//#include <iostream>
//#include <unordered_map>
//#include <unordered_set>
//
//int main(int argc, char** argv){
//    std::string str = "abcdd";
//
//    std::unordered_map<char, std::unordered_set<int>> a;
//    std::unordered_map<char, std::unordered_set<int>>::iterator ref;
//    for(auto i = 0; i < str.size(); ++i){
//        ref = a.find(str[i]);
//        if(ref == a.end()){
//            a.insert({str[i], {i}});
//        }
//        else{
//            ref->second.insert({i});
//        }
//    }
//    std::unordered_map<int, std::unordered_set<int>> l;
//    std::unordered_map<int, std::unordered_set<int>>::iterator l_ref;
//    int min = 0;
//    int now = 0;
//    for(auto i = a.begin(); i != a.end(); ++i){
//        now = i->second.size();
//        min = now < min ? now : min == 0 ? now : min;
//        if(now <= min){
//            l_ref = l.find(now);
//            if(l_ref != l.end()){
//                l_ref->second.insert(i->second.begin(), i->second.end());
//            } 
//            else{
//                l.insert({now, i->second});
//            }
//        }
//        else if(now < min){
//            l.insert({now, i->second});
//        }
//        else{
//            continue;
//        }
//    }
//    auto out = l.find(min)->second;
//    std::string result = "";
//    for(auto i = 0; i < str.size(); ++i){
//        if(out.find(i) == out.end()){
//            result += str[i];
//        }
//    }
//    result = result == "" ? "empty" : result;
//    std::cout << result << std::endl;
//    return 0;
//}


#include <vector>
#include <iostream>

int main(int argc, char** argv){
    std::vector<int> pool = {2, 1, 3, 4, 6, 5, 7, 8, 9, 10};
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
    for(auto p = pool.begin(); p != pool.end(); ++p){
        if(a_amount == 5){
            b_amount += 1;
            b_sum += *p;
            continue;
        }
        else if(b_amount == 5){
            a_amount += 1;
            a_sum += *p;
            continue;
        }
        if(a_sum == b_sum){
            if(a_amount <= b_amount){
                a_amount += 1;
                a_sum += *p;
            }
            else{
                b_amount += 1;
                b_sum += *p;
            }
            continue;
        }
        if(a_sum < b_sum){
            a_sum += *p;
            a_amount += 1; 
            continue;
        }
        if(a_sum > b_sum){
            b_sum += *p;
            b_amount += 1;
            continue;
        }
    }
    int sub = a_sum >= b_sum ? a_sum - b_sum : b_sum - a_sum;
    std::cout << sub << std::endl;
    return 0;
}