/*
 * @lc app=leetcode.cn id=59 lang=cpp
 * @lcpr version=30200
 *
 * [59] 螺旋矩阵 II
 *
 * https://leetcode.cn/problems/spiral-matrix-ii/description/
 *
 * algorithms
 * Medium (70.72%)
 * Likes:    1453
 * Dislikes: 0
 * Total Accepted:    543.7K
 * Total Submissions: 768.9K
 * Testcase Example:  '3'
 *
 * 给你一个正整数 n ，生成一个包含 1 到 n^2 所有元素，且元素按顺时针顺序螺旋排列的 n x n 正方形矩阵 matrix 。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 输入：n = 3
 * 输出：[[1,2,3],[8,9,4],[7,6,5]]
 * 
 * 
 * 示例 2：
 * 
 * 输入：n = 1
 * 输出：[[1]]
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= n <= 20
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    vector<vector<int>> generateMatrix(int n) {
        vector<vector<int>> result(n, vector<int>(n,-1));
        int up = 0, down = n-1;
        int left = 0, right = n-1;
        int cnt = 1;
        while(cnt <= n*n) {
            if (up <= down) {
                for(int i = left; i<=right; i++) {
                    result[up][i] = cnt++;
                }
                up++;
            }
            if(right >= left) {
                for(int i = up; i<=down; i++) {
                    result[i][right] = cnt++;
                }
                right--;
            }

            if(down >= up) {
                for(int i = right; i>=left; i--) {
                    result[down][i] = cnt++;
                }
                down--;
            }

            if(left <= right) {
                for(int i = down; i>=up; i--) {
                    result[i][left] = cnt++;
                }
                left++;
            }
        }
        return result;
    }
};
// @lc code=end



/*
// @lcpr case=start
// 3\n
// @lcpr case=end

// @lcpr case=start
// 1\n
// @lcpr case=end

 */

