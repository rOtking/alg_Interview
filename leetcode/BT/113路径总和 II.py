# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def pathSum1(self, root: TreeNode, targetSum: int) -> List[List[int]]:
        if root is None:
            return []
        if root.left is None and root.right is None:
            return [[root.val]] if root.val == targetSum else []

        lefts = self.pathSum(root.left, targetSum - root.val)
        rights = self.pathSum(root.right, targetSum - root.val)

        res = []
        for path in lefts:
            tmp = path[:]
            tmp.insert(0, root.val)
            res.append(tmp)
        for path in rights:
            tmp = path[:]
            tmp.insert(0, root.val)
            res.append(tmp)

        return res

    # BFS 应该还是遍历所有路径之后得到结果，没什么新奇的，时间也不快，不写了。




# 找到所有路径 从头到尾

# ok
