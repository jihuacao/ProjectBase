#include <boost/container/vector.hpp>
#include "../PR.hpp"

int main(int argc, char** argv){
    auto pre = boost::container::vector<double>({0, 1, 1, 0, 0, 0, 1, 0, 1 , 1, 1, 1});
    auto label = boost::container::vector<double>({0, 1, 1, 0, 0, 0, 1, 0, 1 , 1, 1, 1});
    PR(pre, label);
    return 0;
}