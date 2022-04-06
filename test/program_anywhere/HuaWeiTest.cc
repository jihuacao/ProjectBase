#include <gtest/gtest.h>
#include <climits>
#include <map>

TEST(ProgramAnywhere, HuaWeiTestOne) {
	//int a = 1, b = 20;
	//for (auto i = 0; i < b; ++i) {

	//}
}

bool check_common(int a, int b){
    int remain;
    int temp;
    temp = std::min(a, b);
    a = std::max(a, b);
    b = temp;
    while(true){
        remain = a - b * (a / b);
        if(remain == 0)
            return true;
        else if(remain == 1)
            break;
        else{
            a = b;
            b = remain;
        }
    }
    return false;
};

TEST(ProgramAnywhere, HuaWeiTestTwo) {
    int a = 1, b = 20;
	for(auto o = a; o < b; ++o){
        for(auto t = o + 1; t < b; ++t){
            if(check_common(o, t)){
                continue;
            }
            for(auto th = t + 1; th < b; ++th){
                if(!check_common(t, th) && !check_common(o, th))
                    std::cout << o << " " << t << " " << th << "\n";
            }
        }
    }
}

TEST(ProgramAnywhere, HuaWeiTestThree) {
    std::string s = "asdfghjamns";
    std::map<char, int> start;
    std::map<char, int> end;
    int bb = 0, ee = 0;
    int i = 0;
    for(auto e = s.begin(); e != s.end(); ++e){
        auto found = start.find(*e);
        if(found != start.end()){
            [&](int& _b, int& _e, int now, int t) {
                if (_e - _b < now - t) {
                    _b = t;
                    _e = now;
                }
            }(bb, ee, i, found->second);
        }
        else{
            start.insert({ *e, i });
        }
        i++;
    }
    std::cout << s.substr(bb, ee - bb + 1);
}

//https://leetcode-cn.com/problems/maximum-subarray/
TEST(LeetCode, MaxiMumSubArray) {
    std::vector<int> nums = { -2,1,-3,4,-1,2,1,-5,4 };
    int max_sum = INT_MIN;
    int gap_sum = 0;
    int b = 0, e = 0;
    for (auto iter = nums.begin(); iter != nums.end(); ++iter) {
        if (max_sum > max_sum + gap_sum + *iter) {
            max_sum = max_sum + gap_sum + *iter;
            gap_sum = 0;
        }
        else {
            gap_sum += *iter;
        }
    }
    std::cout << max_sum;
}