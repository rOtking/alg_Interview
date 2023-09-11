# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        # BFS的应用，递归再拼接太麻烦，直接删除了。
        if root is None:
            return []
        queue = [root]
        res = []
        l2r = False
        while(len(queue) != 0):
            size = len(queue)
            l2r = not l2r
            tmp = []
            for _ in range(size):
                if l2r:
                    cur = queue.pop(0)
                    tmp.append(cur.val)
                    if cur.left:
                        queue.append(cur.left)
                    if cur.right:
                        queue.append(cur.right)
                else:
                    cur = queue.pop()
                    tmp.append(cur.val)
                    if cur.right:
                        queue.insert(0, cur.right)
                    if cur.left:
                        queue.insert(0, cur.left)
            res.append(tmp)
        return res

# todo BFS的实现。
# 正向时，头出，子节点加到queue尾不影响本轮正向输出；   输出的queue是从头到尾正向
# 反向时，尾出（后面的先出），右左结点插入queue头，不影响本轮逆向输出；     输出的queue依然是从头到尾正向



# 左到右，右到左

# ok
