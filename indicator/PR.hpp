#include <glog/logging.h>
#include <boost/container/vector.hpp>
#include <boost/timer.hpp>
#include <boost/assert.hpp>
#include <boost/range/algorithm.hpp>
//#include <boost/sort/sample_sort/sample_sort.hpp>
//#include <boost/range/algorithm/transform.hpp>
//#include <boost/algorithm/cxx11/iota.hpp>

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
template <typename _LABEL_TYPE, typename _OUT>
unsigned long long get_label_p_n(const _LABEL_TYPE& label, _OUT* const & p, _OUT* const & n){
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

//template <typename T, typename Compare>
//boost::container::vector<std::size_t> sort_permutation(
//    const boost::container::vector<T>& vec,
//    Compare& compare)
//{
//    boost::container::vector<size_t> p(vec.size());
//    boost::algorithm::iota(p.begin(), p.end(), 0);
//    boost::sort::sample_sort(p.begin(), p.end(), [&](size_t i, size_t j){
//        return vec[i] < vec[j];});
//    return p;
//}
//
//template <typename T>
//std::vector<T> apply_permutation(
//    const std::vector<T>& vec,
//    const std::vector<std::size_t>& p)
//{
//    std::vector<T> sorted_vec(vec.size());
//    std::transform(p.begin(), p.end(), sorted_vec.begin(),
//        [&](std::size_t i){ return vec[i]; });
//    return sorted_vec;
//}
//
//template <typename T>
//void apply_permutation_in_place(
//    std::vector<T>& vec,
//    const std::vector<std::size_t>& p)
//{
//    std::vector<bool> done(vec.size());
//    for (std::size_t i = 0; i < vec.size(); ++i)
//    {
//        if (done[i])
//        {
//            continue;
//        }
//        done[i] = true;
//        std::size_t prev_j = i;
//        std::size_t j = p[i];
//        while (i != j)
//        {
//            std::swap(vec[prev_j], vec[j]);
//            done[j] = true;
//            prev_j = j;
//            j = p[j];
//        }
//    }
//}

template <typename _SCORE_TYPE, typename _LABEL_TYPE>
_SCORE_TYPE PR(const _SCORE_TYPE& score, const _LABEL_TYPE& label){
    int TP = 0;
    int TN = 0;
    int FP = 0;
    int FN = 0;
    unsigned long long p = 0;
    unsigned long long n = 0;
    BOOST_ASSERT(get_label_p_n(score, &p, &n) == NULL);
    //auto index = boost::sort(boost::const_begin(score), boost::const_end(score));//, [&](_SCORE_TYPE::const_iterator i, _SCORE_TYPE::const_iterator j) -> bool { return *i < *j; });
    //DLOG(INFO) << "Done";
    return _SCORE_TYPE();
}