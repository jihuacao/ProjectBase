#include <boost/container/vector.hpp>
#include <boost/timer.hpp>
#include <boost/assert.hpp>

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

unsigned long long get_tp(boost::container::vector<double>& score, boost::container::vector<bool>& label);

/**\brief
 * \note
 * \param[in] label
 * \param[out] p
 * \param[out] n
 * */
template<typename In, typename Out>
unsigned long long get_label_p_n(boost::container::vector<In>& label, Out* const & p, Out* const & n){
    for(auto iter = label.begin(); iter != label.end(); ++iter){
        if(*iter == 1){
            (*p)++;
        }
        else{
            (*n)++;
        }
    }
    return NULL;
}

template<typename T> T p(const T& TP){

}

boost::container::vector<pr> PR(boost::container::vector<double>& score, boost::container::vector<double>& label){
    int TP = 0;
    int TN = 0;
    int FP = 0;
    int FN = 0;
    unsigned long long p = 0;
    unsigned long long n = 0;
    BOOST_ASSERT(get_label_p_n(score, &p, &n) == NULL);
    for(auto index = 0; index < score.size(); ++index){
        if(label[index] == 0){
        }
    }
}