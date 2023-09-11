# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def rightSideView(self, root: TreeNode) -> List[int]:
        # bfs 的每层最右收集起来
        res = []
        if root is None:
            return res
        queue = [root]
        while (len(queue) != 0):
            size = len(queue)
            if size > 0:
                res.append(queue[-1].val)
            for _ in range(size):
                cur = queue.pop(0)
                if cur.left:
                    queue.append(cur.left)
                if cur.right:
                    queue.append(cur.right)

        return res


# ok
# todo BFS yyds
