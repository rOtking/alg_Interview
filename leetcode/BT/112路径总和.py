# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def hasPathSum(self, root: TreeNode, targetSum: int) -> bool:
        if root is None:
            return False
        # 叶子节点是终止条件
        if root.left is None and root.right is None:
            return True if root.val == targetSum else False

        l = self.hasPathSum(root.left, targetSum - root.val)
        r = self.hasPathSum(root.right, targetSum - root.val)

        return l or r

# todo 可以BFS每到一层都记录当前层所有结点的路径和。  简单不写了
# ok
