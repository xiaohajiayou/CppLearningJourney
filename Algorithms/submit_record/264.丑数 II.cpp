/*
 * @lc app=leetcode.cn id=264 lang=cpp
 * @lcpr version=30104
 *
 * [264] 丑数 II
 *
 * https://leetcode.cn/problems/ugly-number-ii/description/
 *
 * algorithms
 * Medium (58.09%)
 * Likes:    1254
 * Dislikes: 0
 * Total Accepted:    198.6K
 * Total Submissions: 342K
 * Testcase Example:  '10'
 *
 * 给你一个整数 n ，请你找出并返回第 n 个 丑数 。
 * 
 * 丑数 就是质因子只包含 2、3 和 5 的正整数。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 输入：n = 10
 * 输出：12
 * 解释：[1, 2, 3, 4, 5, 6, 8, 9, 10, 12] 是由前 10 个丑数组成的序列。
 * 
 * 
 * 示例 2：
 * 
 * 输入：n = 1
 * 输出：1
 * 解释：1 通常被视为丑数。
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= n <= 1690
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    int nthUglyNumber(int n) {
        int p2 = 1,p3 =1,p5 = 1;
        int pr2 = 1,pr3 =1,pr5 = 1;
        vector<int> vec(n+1);
        int p = 1;
        while(p<=n) {
            int min_v = min({pr2,pr3,pr5});
            vec[p]=min_v;
            p++;
            if(pr2==min_v) {
                pr2 = vec[p2]*2;
                p2++;
            } 
            if(pr3==min_v) {
                pr3 = vec[p3]*3;
                p3++;
            }
            if(pr5==min_v) {
                pr5 = vec[p5]*5;
                p5++;
            }
        }
        return vec[n];
    }
};
// @lc code=end



/*
// @lcpr case=start
// 10\n
// @lcpr case=end

// @lcpr case=start
// 1\n
// @lcpr case=end

 */

