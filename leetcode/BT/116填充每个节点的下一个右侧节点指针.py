# Definition for a Node.
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next


class Solution:
    # 递归
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
        # 左节点的右边界与右结点的左边界连接
        while (left.right and right.left):
            left = left.right
            right = right.left
            left.next = right
        return

    # 基于BFS的层次遍历
    def connect(self, root: 'Node') -> 'Node':
        if root is None:
            return root
        queue = [root]
        while (len(queue) != 0):
            size = len(queue)
            pre = None
            for _ in range(size):
                cur = queue.pop(0)
                if pre is not None:
                    pre.next = cur
                pre = cur
                if cur.left:
                    queue.append(cur.left)
                if cur.right:
                    queue.append(cur.right)
        return root


# ok
# 注意不同根节点的连接！其实就是对于递归函数的定义要清楚：是相邻节点，不论是不是同一个根节点的，只要相邻就相连，不要忘了。
# 在大递归用的时候直接用现成的即可，不要陷入细节！