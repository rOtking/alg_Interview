# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    # todo 代码简单好理解！这就是DFS啊！！！别不认识了
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        if p is None and q is None:
            return True
        if p is None or q is None:
            return False

        return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
    # 尝试一下 DFS与BFS

    # BFS版本
    def isSameTree2(self, p: TreeNode, q: TreeNode) -> bool:
        if p is None and q is None:
            return True
        if p is None or q is None:
            return False
        queue_p = [p]
        queue_q = [q]
        while (len(queue_p) != 0 and len(queue_q) != 0):
            cur_p = queue_p.pop(0)
            cur_q = queue_q.pop(0)

            if cur_p or cur_q:
                if cur_p is None or cur_q is None or cur_p.val != cur_q.val:
                    return False

                queue_p.append(cur_p.left)
                queue_p.append(cur_p.right)
                queue_q.append(cur_q.left)
                queue_q.append(cur_q.right)
        return True

    # 最开始就是DFS发现了么，一直走left走到底，才去右边！
    # def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
    #     def backtrack(track, candidates):
    #         pass

    #     return backtrack()
# ok
