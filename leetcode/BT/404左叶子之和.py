# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sumOfLeftLeaves(self, root: TreeNode) -> int:
        return self.process(root, False)

    def process(self, root, isLeft):
        if root is None:
            return 0
        if root.left is None and root.right is None and isLeft:
            return root.val

        res1 = self.process(root.left, True)
        res2 = self.process(root.right, False)

        return res1 + res2



# todo 一道怪怪的简单题，做出来了，再反复看看。