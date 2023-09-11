# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    # 不能只看左右等不等，是左子树与右子树是不是全等！！！
    def countUnivalSubtreesERROR(self, root: TreeNode) -> int:
        if root is None:
            return 0
        if root.left is None and root.right is None:
            return 1

        res1 = self.countUnivalSubtrees(root.left)
        res2 = self.countUnivalSubtrees(root.right)
        if root.left is None:
            return res2 + 1 if root.val == root.right.val else res2
        if root.right is None:
            return res1 + 1 if root.val == root.left.val else res1
        return res1 + res2 + 1 if root.val == root.left.val and root.val == root.right.val else res1 + res2

    def countUnivalSubtrees(self, root: TreeNode) -> int:
        def process(root):
            # 返回个数与自己本身是不是
            if root is None:
                return 0, False
            if root.left is None and root.right is None:
                return 1, True

            leftNum, isLeft = process(root.left)
            rightNum, isRight = process(root.right)

            if root.left is None:
                return (rightNum + 1, True) if isRight and root.val == root.right.val else (rightNum, False)
            if root.right is None:
                return (leftNum + 1, True) if isLeft and root.val == root.left.val else (leftNum, False)

            return (leftNum + rightNum + 1,
                    True) if isLeft and isRight and root.val == root.left.val and root.val == root.right.val else (
            leftNum + rightNum, False)

        return process(root)[0]

        # 有个坑啊！这种返回元祖的方式得加括号啊，不然逻辑就变了！！！


