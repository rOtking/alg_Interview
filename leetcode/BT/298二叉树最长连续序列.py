# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def __init__(self) -> None:
        self.res = 0

    def longestConsecutive(self, root: TreeNode) -> int:
        pass
    def process(self, root):
        # dfs
        if root is None:
            return 0
        if root.left is None and root.right is None:
            return 1

        res1 = self.process(root.left)
        res2 = self.process(root.right)




# todo 太难跳过
