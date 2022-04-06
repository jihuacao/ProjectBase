#include <functional>
#include <gtest/gtest.h>

std::function<int(int&)> generate_lambda_function(int t1){
    return [&](int& t2) -> int{
        std::swap(t1, t2);
        return 0;
    };
};

class Te{
    public:
        Te(const Te& lref){
            _a = lref._a;
            _b = lref._b;
        };
        Te(){
            _a = 0;
            _b = 0;
        };
        Te(Te&& rref){
            std::swap(_a, rref._a);
            std::swap(_b, rref._b);
        };
    public:
        void operator=(Te&& rref){
            std::swap(_a, rref._a);
            std::swap(_b, rref._b);
        }
    public:
        int _a;
        float _b;
};

int test(int t1, Te te, Te& rete){
    te._a = te._a + 1;
    rete = std::move(te);
    return t1;
};

TEST(B, B){
    int t1 = 1;
    int t2 = 2;
    Te inpt;
    Te oupt;
    auto t3 = test(t1, inpt, oupt);
    auto func = generate_lambda_function(t1);
    ASSERT_EQ(func(t2), 0);
}