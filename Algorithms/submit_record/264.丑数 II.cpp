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
        vector<int> ugly(n+1,1) ;
        int p2 = 0, p3 = 0, p5 = 0;
        int p = 1;
        for(int i = 0; i<n; i++) {

            int mini = min({ugly[p2]*2,ugly[p3]*3,ugly[p5]*5});
            ugly[p] = mini;
            p++;
            if(mini == ugly[p2]*2) p2++;
            if(mini == ugly[p3]*3) p3++;
            if(mini == ugly[p5]*5) p5++;
        }

        return ugly[n-1];

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

