# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if root is None:
            return []

        res = [[root.val]]
        lefts = self.levelOrder(root.left)
        rights = self.levelOrder(root.right)
        i = 0
        while (i < len(lefts) and i < len(rights)):
            tmp = lefts[i]
            tmp.extend(rights[i])
            res.append(tmp)
            i += 1
        if i < len(lefts):
            res.extend(lefts[i:])
        if i < len(rights):
            res.extend(rights[i:])

        return res

    # todo 天生就是BFS啊！！！

    def levelOrder1(self, root: TreeNode) -> List[List[int]]:
        if root is None:
            return []
        res = []
        queue = [root]
        while (len(queue) != 0):
            size = len(queue)
            tmp = []
            for _ in range(size):
                cur = queue.pop(0)
                tmp.append(cur.val)
                if cur.left:
                    queue.append(cur.left)
                if cur.right:
                    queue.append(cur.right)

            res.append(tmp)

        return res

# ok
# 递归很容易解决