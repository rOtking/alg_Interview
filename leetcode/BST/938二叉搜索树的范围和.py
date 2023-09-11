# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def rangeSumBST(self, root: TreeNode, low: int, high: int) -> int:
        if root is None:
            return 0

        res = 0
        if root.val < low:
            res += self.rangeSumBST(root.right, low, high)
        elif root.val > high:
            res += self.rangeSumBST(root.left, low, high)
        else:
            res += self.rangeSumBST(root.left, low, high)
            res += self.rangeSumBST(root.right, low, high)
            res += root.val

        return res

# ok
# easy