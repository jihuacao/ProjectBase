#include <string>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <gtest/gtest.h>

TEST(DropTheLeast, Int){
    std::string str = "abcdd";

    std::unordered_map<char, std::unordered_set<int>> a;
    std::unordered_map<char, std::unordered_set<int>>::iterator ref;
    for(auto i = 0; i < str.size(); ++i){
        ref = a.find(str[i]);
        if(ref == a.end()){
            a.insert({str[i], {i}});
        }
        else{
            ref->second.insert({i});
        }
    }
    std::unordered_map<int, std::unordered_set<int>> l;
    std::unordered_map<int, std::unordered_set<int>>::iterator l_ref;
    int min = 0;
    int now = 0;
    for(auto i = a.begin(); i != a.end(); ++i){
        now = i->second.size();
        min = now < min ? now : min == 0 ? now : min;
        if(now <= min){
            l_ref = l.find(now);
            if(l_ref != l.end()){
                l_ref->second.insert(i->second.begin(), i->second.end());
            } 
            else{
                l.insert({now, i->second});
            }
        }
        else if(now < min){
            l.insert({now, i->second});
        }
        else{
            continue;
        }
    }
    auto out = l.find(min)->second;
    std::string result = "";
    for(auto i = 0; i < str.size(); ++i){
        if(out.find(i) == out.end()){
            result += str[i];
        }
    }
    result = result == "" ? "empty" : result;
    std::cout << result << std::endl;
}
