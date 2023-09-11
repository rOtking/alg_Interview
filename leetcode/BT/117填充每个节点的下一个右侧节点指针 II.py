"""
# Definition for a Node.
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""


class Solution:
    def connect1(self, root: 'Node') -> 'Node':
        if root is None:
            return root

        self.connectTwoNode(root.left, root.right)
        return root

    def connectTwoNode(self, left, right):
        '''
        将left子树与right子树所有相邻的节点都连起来
        :param left:
        :param right:
        :return:
        '''
        if left is None or right is None:
            return

        left.next = right
        # 左内部
        self.connectTwoNode(left.left, left.right)
        # 右内部
        self.connectTwoNode(right.left, right.right)
        # 左结点最右边界连右结点最左边界

        p1 = left
        p2 = right
        while (p1 and p2):
            p1.next = p2
            p1 = p1.right if p1.right else p1.left
            p2 = p2.left if p2.left else p2.right
        # 存在最左边界没有次左边界长的情况，这是需要次左边界连接了！这种问题比较难处理，所以这种方法不合适！
        return

    # 这种题，BFS yyds！
    def connect(self, root: 'Node') -> 'Node':
        if root is None:
            return root
        queue = [root]

        while (len(queue) != 0):
            size = len(queue)
            pre = None
            for _ in range(size):
                cur = queue.pop(0)
                if pre:
                    pre.next = cur
                pre = cur
                if cur.left:
                    queue.append(cur.left)
                if cur.right:
                    queue.append(cur.right)
        return root

# BFS 包打一切！哈哈哈

