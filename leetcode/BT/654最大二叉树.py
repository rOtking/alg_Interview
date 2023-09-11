# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def constructMaximumBinaryTree(self, nums: List[int]) -> TreeNode:

        if len(nums) == 0:
            return None

        max_value = max(nums)
        max_index = nums.index(max_value)
        # 后序遍历
        left = self.constructMaximumBinaryTree(nums[:max_index])
        right = self.constructMaximumBinaryTree(nums[max_index + 1:])

        # 构造根节点
        root = TreeNode(max_value, left=left, right=right)

        return root



# ok
# 居然这么简单就过了，果然不能陷入细节。
# 感觉其实还不清楚细节，就已经结束了！！hahaha   体会这种感觉
