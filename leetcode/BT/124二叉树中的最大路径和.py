# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def __init__(self):
        self.res = float('-inf')
    def maxPathSum(self, root: TreeNode) -> int:
        self.maxSum(root)
        return self.res

    def maxSum(self, root):
        # 关键！
        if root is None:
            return 0
        # 可省
        # if root.left is None and root.right is None:
        #     self.res = root.val
        #     return root.val

        # 如果子树是负数，也就是对当前节点的总和没有贡献，那就不要子树
        left = max(self.maxSum(root.left), 0)
        right = max(self.maxSum(root.right), 0)

        # 当前节点为根的最大值
        cur = left + right + root.val
        self.res = cur if cur > self.res else self.res
        # 记录时：记录根节点 + 左右
        # 返回时：只能返回根节点 + 最大的一个子树。因为父节点要的子路径时单链！
        # todo 关键
        return root.val + max(left, right)





# 一时没思路，看答案了。
# todo 重点看：换个问法：以某个节点为根节点的任意路径和最大！
# todo 不管是什么路径，都一定有一个根节点！这就是核心
# 搞个全局变量更新最大值

# ok 但要重点看！