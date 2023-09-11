# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    # 这就是DFS，递归就是一种回溯！
    # 要认得这就是DFS
    def maxDepth1(self, root: TreeNode) -> int:

        if root is None:
            return 0
        if root.left is None and root.right is None:
            return 1

        left_max = self.maxDepth(root.left)
        right_max = self.maxDepth(root.right)

        return max(left_max, right_max) + 1

    # 当然可以BFS数层数
    def maxDepth(self, root: TreeNode) -> int:
        if root is None:
            return 0

        queue = [root]
        res = 0
        while (len(queue) != 0):
            size = len(queue)
            for _ in range(size):
                cur = queue.pop(0)
                if cur.left:
                    queue.append(cur.left)
                if cur.right:
                    queue.append(cur.right)
            res += 1
        return res

# ok
