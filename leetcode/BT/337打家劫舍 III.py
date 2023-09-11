# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def __init__(self) -> None:
        self.memo1 = {}
        self.memo2 = {}

    def rob1(self, root: TreeNode) -> int:
        if root is None:
            return 0

        money = root.val
        if root.left is not None:
            money += self.rob(root.left.left)
            money += self.rob(root.left.right)
        if root.right is not None:
            money += self.rob(root.right.left)
            money += self.rob(root.right.right)

        # 两种情况：有root与没有root
        # money就是有root的值，与没有的比一下即可
        return max(money, self.rob(root.left) + self.rob(root.right))



    def rob(self, root: TreeNode) -> int:
        if root is None:
            return 0
        return max(self.process1(root), self.process2(root))

    def process1(self, root):
        # 选择当前结点获取的最大值
        if root is None:
            return 0
        if root.left not in self.memo2:
            self.memo2[root.left] = self.process2(root.left)
        if root.right not in self.memo2:
            self.memo2[root.right] = self.process2(root.right)
        return self.memo2[root.left] + self.memo2[root.right] + root.val

    def process2(self, root):
        # 不选当前结点的最大值
        if root is None:
            return 0
        if root.left not in self.memo1:
            self.memo1[root.left] = self.process1(root.left)
        if root.left not in self.memo2:
            self.memo2[root.left] = self.process2(root.left)
        if root.right not in self.memo1:
            self.memo1[root.right] = self.process1(root.right)
        if root.right not in self.memo2:
            self.memo2[root.right] = self.process2(root.right)
        return max(self.memo1[root.left], self.memo2[root.left]) + max(self.memo1[root.right], self.memo2[root.right])

# todo 树型DP的递归思路有了，但是BT写DP表不知道怎么办。memo先提速。