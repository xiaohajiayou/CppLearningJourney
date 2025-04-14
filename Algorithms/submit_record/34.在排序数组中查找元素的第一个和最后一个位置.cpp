/*
 * @lc app=leetcode.cn id=34 lang=cpp
 * @lcpr version=30104
 *
 * [34] 在排序数组中查找元素的第一个和最后一个位置
 */

// @lc code=start
class Solution {
public:
    int findLeft(vector<int>&nums, int target) {
        int i = 0,j = nums.size();
        while(i<j) {
            int mid = i+(j-i)/2;
            if(nums[mid]<target) {
                i = mid +1;
            } else if(nums[mid]>target) {
                j = mid;
            } else if(nums[mid] == target) {
                j = mid;
            }
        }
        if(i<0||j>=nums.size()||nums[i]!=target) return -1;
        return i;
    }
    int findRight(vector<int>&nums, int target) {
        int i = 0,j = nums.size();
        while(i<j) {
            int mid = i+(j-i)/2;
            if(nums[mid]<target) {
                i = mid+1;
            } else if(nums[mid]>target) {
                j = mid;
            } else if(nums[mid] == target) {
                i = mid+1;
            }
        }
        if((j)<0 || (j-1)>=nums.size() ||nums[j-1]!=target) return -1;
        return j-1;
    }
    vector<int> searchRange(vector<int>& nums, int target) {
        int left = findLeft(nums,target);
        int right = findRight(nums,target);
        return {left,right};
    }
};
// @lc code=end



/*
// @lcpr case=start
// [5,7,7,8,8,10]\n8\n
// @lcpr case=end

// @lcpr case=start
// [5,7,7,8,8,10]\n6\n
// @lcpr case=end

// @lcpr case=start
// []\n0\n
// @lcpr case=end

 */

