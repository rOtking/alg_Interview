# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        if root is None:
            return True
        flag = self.check(root.left, root.right)

        return flag

    def check(self, p, q):
        '''
        判断两棵树属否互为对称
        :param p:
        :param q:
        :return:
        '''
        if p is None and q is None:
            return True
        if p is None or q is None:
            return False
        # 下面都是非空
        flag1 = self.check(p.left, q.right)
        flag2 = self.check(p.right, q.left)
        flag3 = True if p.val == q.val else False
        return flag1 and flag2 and flag3


    # BFS
    def isSymmetricBFS(self, root: TreeNode) -> bool:
        if root is None:
            return True
        queue = [root]
        while(len(queue) != 0):
            res = []
            size = len(queue)
            for _ in range(size):
                cur = queue.pop(0)
                res.append(cur.left.val if cur.left else '#')
                res.append(cur.right.val if cur.right else '#')
                if cur.left:
                    queue.append(cur.left)
                if cur.right:
                    queue.append(cur.right)
            # 判断回文
            i, j = 0, len(res) - 1
            while(i <= j):
                if res[i] != res[j]:
                    return False
                i += 1
                j -= 1
        return True




# todo 这个重点看！已知没想出来，其实很简单。确实还是递归，但是要重新定义函数：
# todo 判断两棵树互为镜像！
# 一定要会转化问题，当前的大函数是不能完成递归的，那就要进一步转化问题，自己定义新的函数。

# 注意：不需要任意子树都对称，总体对称就行了！

