# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def binaryTreePaths(self, root: TreeNode) -> List[str]:
        if root is None:
            return []
        if root.left is None and root.right is None:
            return [str(root.val)]

        lefts = self.binaryTreePaths(root.left)
        rights = self.binaryTreePaths(root.right)

        tmp = lefts
        tmp.extend(rights)
        res = []
        for ele in tmp:
            res.append(str(root.val) + '->' + ele)
        return res

# ok
