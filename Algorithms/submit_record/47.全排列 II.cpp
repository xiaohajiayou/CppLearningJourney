/*
 * @lc app=leetcode.cn id=47 lang=cpp
 * @lcpr version=30104
 *
 * [47] 全排列 II
 *
 * https://leetcode.cn/problems/permutations-ii/description/
 *
 * algorithms
 * Medium (66.65%)
 * Likes:    1702
 * Dislikes: 0
 * Total Accepted:    653.8K
 * Total Submissions: 980.9K
 * Testcase Example:  '[1,1,2]'
 *
 * 给定一个可包含重复数字的序列 nums ，按任意顺序 返回所有不重复的全排列。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 输入：nums = [1,1,2]
 * 输出：
 * [[1,1,2],
 * ⁠[1,2,1],
 * ⁠[2,1,1]]
 * 
 * 
 * 示例 2：
 * 
 * 输入：nums = [1,2,3]
 * 输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= nums.length <= 8
 * -10 <= nums[i] <= 10
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    void backtrace(vector<int>& nums , vector<int> used) {
        if(path.size() == nums.size()) {
            result.push_back(path);
            return ;
        }

        for(int i = 0;i<nums.size();i++) {
            if(used[i]!=-1) continue;
            if(i>0 && used[i-1]==-1 && nums[i-1] == nums[i]) continue;
            used[i] = 1;
            path.push_back(nums[i]);
            backtrace(nums,used);
            path.pop_back();
            used[i] = -1;
        }
        
    }
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        vector<int> used(nums.size(),-1);
        backtrace(nums,used);
        return result;
    }

private:
    vector<int> path;
    vector<vector<int>> result;
};
// @lc code=end



/*
// @lcpr case=start
// [1,1,2]\n
// @lcpr case=end

// @lcpr case=start
// [1,2,3]\n
// @lcpr case=end

 */

