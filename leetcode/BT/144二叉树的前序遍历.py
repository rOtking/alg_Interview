# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    # todo 垃圾方法，看左神的！！！！哈哈
    def preorderTraversal1(self, root: TreeNode) -> List[int]:
        res = []   # 存结果
        if root is None:
            return []
        stack = []  # 保存父节点
        cur = root  # 当前节点

        # stack空表示就是在主栈，还没入栈或者已经pop回主栈了
        # cur为None就是遍历到None的叶子了，不代表完全结束；但需要加他进行初始化，不然进不去循环
        while(len(stack) != 0 or cur is not None):
            while(cur is not None):
                res.append(cur.val)
                stack.append(cur)
                cur = cur.left
            # 左空时跳出，找父的有节点继续，他就是root了
            cur = stack.pop()   # 左空的父节点
            cur = cur.right

        return res

    def preorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        if not root:
            return res
        stack = [root]
        while(len(stack) != 0):
            cur = stack.pop()
            res.append(cur.val)
            if cur.right:
                stack.append(cur.right)
            if cur.left:
                stack.append(cur.left)

        return res


# ok
# todo 重点看！迭代是怎么保证不遗漏的！也就是循环的条件设置很巧！





# 迭代解法