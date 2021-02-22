#include <vector>

struct pr{
    float percision;
    float recall;
};

struct pn{
    unsigned long long TP;
    unsigned long long TN;
    unsigned long long FP;
    unsigned long long FN;
};

unsigned long long get_tp(std::vector<double>& score, std::vector<bool>& label){
}

std::vector<pr> PR(std::vector<double>& score, std::vector<bool>& label){
    int TP = ;
    int TN = 0;
    int FP = ;
    int FN = 0;
    for(auto index = 0; index < score.size(); ++index){
        if(label[index] == 0){
        }
    }
}