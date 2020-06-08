#include <iostream>

int main(int argc, char** argv){
    bool one = true;
    bool two = false;
    auto t = (one &= two);
    std::cout << "one: " << one << "; " << "two: " << two << "; " << "t: " << t << std::endl;
}