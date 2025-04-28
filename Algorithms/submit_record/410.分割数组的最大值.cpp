/*
 * @lc app=leetcode.cn id=410 lang=cpp
 * @lcpr version=30104
 *
 * [410] 分割数组的最大值
 *
 * https://leetcode.cn/problems/split-array-largest-sum/description/
 *
 * algorithms
 * Hard (60.67%)
 * Likes:    1046
 * Dislikes: 0
 * Total Accepted:    103K
 * Total Submissions: 169.8K
 * Testcase Example:  '[7,2,5,10,8]\n2'
 *
 * 给定一个非负整数数组 nums 和一个整数 k ，你需要将这个数组分成 k 个非空的连续子数组，使得这 k 个子数组各自和的最大值 最小。
 * 
 * 返回分割后最小的和的最大值。
 * 
 * 子数组 是数组中连续的部份。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 输入：nums = [7,2,5,10,8], k = 2
 * 输出：18
 * 解释：
 * 一共有四种方法将 nums 分割为 2 个子数组。 
 * 其中最好的方式是将其分为 [7,2,5] 和 [10,8] 。
 * 因为此时这两个子数组各自的和的最大值为18，在所有情况中最小。
 * 
 * 示例 2：
 * 
 * 输入：nums = [1,2,3,4,5], k = 2
 * 输出：9
 * 
 * 
 * 示例 3：
 * 
 * 输入：nums = [1,4,4], k = 3
 * 输出：4
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= nums.length <= 1000
 * 0 <= nums[i] <= 10^6
 * 1 <= k <= min(50, nums.length)
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    int fun(vector<int>& nums, int cap) {
        // cout<<cap<<endl;
        int tmp = cap;
        int cnt = 0;
        for(int i = 0; i<nums.size(); i++) {
            tmp-=nums[i];
            if (tmp<0) {
                cnt++;
                tmp = cap-nums[i];
            } else if(tmp == 0) {
                cnt++;
                tmp = cap;
            }
        }
        if(tmp<cap) cnt++;
        return cnt;
    }
    int splitArray(vector<int>& nums, int k) {
        int left = 0, right = 0;
        for(auto c:nums) {
            left = max(left,c);
            right+=c;
        }
        while(left<right) {
            int mid = left+(right-left)/2;
            int tmp = fun(nums,mid);
            if(tmp<=k) {
                right = mid;
            } else if(tmp>k) {
                left = mid+1;
            }
        }
        return left;
    }
};
// @lc code=end



/*
// @lcpr case=start
// [7,2,5,10,8]\n2\n
// @lcpr case=end

// @lcpr case=start
// [1,2,3,4,5]\n2\n
// @lcpr case=end

// @lcpr case=start
// [1,4,4]\n3\n
// @lcpr case=end

 */

