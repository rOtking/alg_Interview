# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    # todo 完全没必要分开，一个函数多个返回值即可
    # def isBalanced(self, root: TreeNode) -> bool:
    #     if root is None:
    #         return True
    #     if root.left is None and root.right is None:
    #         return True

    #     left_flag = self.isBalanced(root.left)
    #     right_flag = self.isBalanced(root.right)
    #     if not left_flag or not right_flag:
    #         return False

    #     left_deepth = self.getDeepth(root.left)
    #     right_deepth = self.getDeepth(root.right)

    #     flag = True if -1 <= left_deepth - right_deepth <= 1 else False

    #     return flag

    # def getDeepth(self, root):
    #     if root is None:
    #         return 0
    #     if root.left is None and root.right is None:
    #         return 1

    #     left_deepth = self.getDeepth(root.left)
    #     right_deepth = self.getDeepth(root.right)

    #     return max(left_deepth, right_deepth) + 1
    def isBalanced(self, root: TreeNode) -> bool:

        return self.process(root)[0]

    # 返回是不是 + 深度
    def process(self, root):
        if root is None:
            return True, 0

        res1 = self.process(root.left)
        res2 = self.process(root.right)

        isOk = False
        if res1[0] and res2[0] and (-2 < res1[1] - res2[1] < 2):
            isOk = True

        height = max(res1[1], res2[1]) + 1

        return isOk, height

# ok
# 善于抽象功能，完成大框架就问题不大！