# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def __init__(self):
        self.helps = []
    def minDiffInBST(self, root: TreeNode) -> int:
        self.inOrder(root)

        min_diff = float('inf')

        for index in range(len(self.helps) - 1):
            if self.helps[index + 1] - self.helps[index] < min_diff:
                min_diff = self.helps[index + 1] - self.helps[index]

        return min_diff


    def inOrder(self, root):
        if root is None:
            return root

        self.inOrder(root.left)
        self.helps.append(root.val)
        self.inOrder(root.right)

        return

# ok   easy