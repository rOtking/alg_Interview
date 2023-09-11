import math
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def countNodes(self, root: TreeNode) -> int:
        if root is None:
            return 0

        l, r = root, root
        hl, hr = 0, 0
        while(l is not None):
            l = l.left
            hl += 1
        while(r is not None):
            r = r.right
            hr += 1
        # 满BT
        if hl == hr:
            return int(math.pow(2, hl)) - 1
        else:
            return 1 + self.countNodes(root.left) + self.countNodes(root.right)

    #递归
    def countNodes1(self, root: TreeNode) -> int:
        if root is None:
            return 0

        return 1 + self.countNodes(root.left) + self.countNodes(root.right)

# BT是O（n），满二叉树是O(logn)，也就是求高度；完全二叉树是中间状态。
# 利用完全二叉树必有一个子树是满二叉树，利用满二叉树的数学性质来加速！

# ok
