/*
 * @lc app=leetcode.cn id=21 lang=cpp
 * @lcpr version=30104
 *
 * [21] 合并两个有序链表
 */

// @lc code=start
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        ListNode* dumy = new ListNode(0);
        ListNode* cur = dumy;
        while(list1&&list2) {
            if(list1->val<list2->val) {
                auto tmp = list1->next;
                cur->next = list1;
                list1->next = nullptr;
                list1 = tmp;
            } else {
                auto tmp = list2->next;
                cur->next = list2;
                list2->next = nullptr;
                list2 = tmp;
            }
            cur = cur->next;
        }
        if(list1) cur->next = list1;
        if(list2) cur->next = list2;
        return dumy->next;
    }
};
// @lc code=end



/*
// @lcpr case=start
// [1,2,4]\n[1,3,4]\n
// @lcpr case=end

// @lcpr case=start
// []\n[]\n
// @lcpr case=end

// @lcpr case=start
// []\n[0]\n
// @lcpr case=end

 */

