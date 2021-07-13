#include <gtest/gtest.h>
#include <vector>

/**
 * 1387. 将整数按权重排序
 * 我们将整数 x 的 权重 定义为按照下述规则将 x 变成 1 所需要的步数：
 * 
 * 如果 x 是偶数，那么 x = x / 2
 * 如果 x 是奇数，那么 x = 3 * x + 1
 * 比方说，x=3 的权重为 7 。因为 3 需要 7 步变成 1 （3 --> 10 --> 5 --> 16 --> 8 --> 4 --> 2 --> 1）。
 * 
 * 给你三个整数 lo， hi 和 k 。你的任务是将区间 [lo, hi] 之间的整数按照它们的权重 升序排序 ，如果大于等于 2 个整数有 相同 的权重，那么按照数字自身的数值 升序排序 。
 * 
 * 请你返回区间 [lo, hi] 之间的整数按权重排序后的第 k 个数。
 * 
 * 注意，题目保证对于任意整数 x （lo <= x <= hi） ，它变成 1 所需要的步数是一个 32 位有符号整数。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 输入：lo = 12, hi = 15, k = 2
 * 输出：13
 * 解释：12 的权重为 9（12 --> 6 --> 3 --> 10 --> 5 --> 16 --> 8 --> 4 --> 2 --> 1）
 * 13 的权重为 9
 * 14 的权重为 17
 * 15 的权重为 17
 * 区间内的数按权重排序以后的结果为 [12,13,14,15] 。对于 k = 2 ，答案是第二个整数也就是 13 。
 * 注意，12 和 13 有相同的权重，所以我们按照它们本身升序排序。14 和 15 同理。
 * */
int getKth(int lo, int hi, int k){
    std::map<int, int> w;
    std::vector<int> wt;
    std::vector<int> mid;
    int size;
    int temp;
    w.insert({1, 0});
    for(auto i = lo; i <= hi; ++i){
        temp = i;
        do{
            if(w.find(temp) != w.end()){
                mid.push_back(w.find(temp)->second);
                break;
            }else{
                mid.push_back(temp);
                if((temp / 2) * 2 == temp){
                    temp = temp / 2;
                }else{
                    temp = 3 * temp + 1;
                }
            }
        }while(temp != 1);
        if(temp == 1){
            mid.push_back(0);
        }
        size = mid.size();
        wt.push_back(i);
        for(auto t = 0; t < size - 1; ++t){
            w.insert({mid[t], *mid.rbegin() + size - 1 - t});
        }
        mid.clear();
    }
    for(auto i = w.begin(); i != w.end(); ){
        if(lo <= i->first && i->first <= hi){
            ++i;
            continue;
        }else{
            w.erase(i++);
        }
    }
    std::sort(wt.begin(), wt.end(), [&](int i, int j){
        if (w[i] != w[j]){
            return w[i] < w[j];
        }
        else{
            return i < j;
        }
    });
    return wt[k - 1];
}

TEST(OneThreeEightSeven, single_test){
    getKth(12, 15, 2);
}

/*
一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。

问总共有多少条不同的路径？
示例 1：
输入：m = 3, n = 7
输出：28

可行解法：
1.动态规划
2.组合数学

来源：力扣（LeetCode）
链接：```https://leetcode-cn.com/problems/unique-paths/```
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
*/
int get(int m, int n, int* ptr, int m_now, int n_now){
    int temp = 0;
    if(ptr[m_now * n + n_now] == -1){
        if(m_now - 1 >= 0){
            temp += get(m, n, ptr, m_now - 1, n_now);
        }
        if(n_now - 1 >= 0){
            temp += get(m, n, ptr, m_now, n_now - 1);
        }
        ptr[m_now * n + n_now] = temp;
        return temp;
    }
    else{
        return ptr[m_now * n + n_now];
    }
};

int uniquePaths(int m, int n) {
    int* ptr = (int*)malloc(sizeof(int) * m * n);
    memset(ptr, -1, sizeof(int) * m * n);
    ptr[0] = 1;
    int temp = 0;
    temp = get(m, n, ptr, m - 1, n - 1);
    free(ptr);
    return temp;
}

TEST(SixTwo, single_test){
    uniquePaths(3, 7);
}

/*
64. 最小路径和
给定一个包含非负整数的 m x n 网格 grid ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。
说明：每次只能向下或者向右移动一步。

示例 1：
输入：grid = [[1,3,1],[1,5,1],[4,2,1]]
输出：7
解释：因为路径 1→3→1→1→1 的总和最小。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/minimum-path-sum
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。*/
/**
 * \brief brief
 * \note note
 *  f(m, n)=v(m, n)+max(f(m-1, n), f(m, n-1))
 * \author none
 * \param[in] in
 * \param[out] out
 * \return return
 * \retval retval
 * \since version
 * */
int get_min_path(int* ptr, std::vector<std::vector<int>>& grid, int m, int n, int m_now, int n_now){
    int temp;
    if(ptr[m_now * m + n_now] == -1){
        if(m_now - 1 >= 0 && n_now - 1 >= 0){
            temp = std::min(get_min_path(ptr, grid, m, n, m_now, n_now - 1), get_min_path(ptr, grid, m, n, m_now - 1, n_now)) + grid[m_now][n_now];
        }
        else{
            if(m_now - 1 < 0){
                temp = get_min_path(ptr, grid, m, n, m_now, n_now - 1) + grid[m_now][n_now];
            }
            if(n_now - 1 < 0){
                temp = get_min_path(ptr, grid, m, n, m_now - 1, n_now) + grid[m_now][n_now];
            }
        }
        ptr[m_now * m + n_now] = temp;
        return temp;
    }
    else{
        return ptr[m_now * m + n_now];
    }
}
int minPathSum(std::vector<std::vector<int>>& grid) {
    int m = grid.size();
    int n = grid[0].size();
    int ret = 0;
    int* ptr = (int*)malloc(m * n * sizeof(int));
    memset(ptr, -1, m * n * sizeof(int));
    ptr[0] = grid[0][0];
    ret = get_min_path(ptr, grid, m, n, m - 1, n - 1);
    free(ptr);
    return ret;
}

TEST(SixFour, single_test){
    std::vector<std::vector<int>> grid;
    grid.push_back(std::vector<int>({1, 2, 3}));
    grid.push_back(std::vector<int>({4, 5, 6}));
    minPathSum(grid);
}

/*
1191. K 次串联后最大子数组之和
给你一个整数数组 arr 和一个整数 k。

首先，我们要对该数组进行修改，即把原数组 arr 重复 k 次。

举个例子，如果 arr = [1, 2] 且 k = 3，那么修改后的数组就是 [1, 2, 1, 2, 1, 2]。

然后，请你返回修改后的数组中的最大的子数组之和。

注意，子数组长度可以是 0，在这种情况下它的总和也是 0。

由于 结果可能会很大，所以需要 模（mod） 10^9 + 7 后再返回。 

示例 1：

输入：arr = [1,2], k = 3
输出：9
示例 2：

输入：arr = [1,-2,1], k = 5
输出：2
示例 3：

输入：arr = [-1,-2], k = 7
输出：0
*/
int kConcatenationMaxSum(std::vector<int>& arr, int k) {
    int p = 0;
    int m = 0;
    int s = 0;
    int l = k >= 3 ? 3 : k;
    for(auto i = 0; i < l * arr.size(); ++i){
        p = std::max(p + arr[i % arr.size()], arr[i % arr.size()]);
        m = std::max(p, m);
        if(i < arr.size()){
            s += arr[i];
        }
    }
    m = std::max(p, m);
    m = std::max(k * s, m);
    return m;
}

TEST(OneOneNineOne, signle_test){
    auto a = std::vector<int>({-5,-2,0,0,3,9,-2,-5,4});
    auto k = 5;
    kConcatenationMaxSum(a, k);
}