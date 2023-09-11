# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def deleteNode(self, root: TreeNode, key: int) -> TreeNode:
        if root is None:
            return None

        if root.val == key:
            # 叶子，直接删了
            if root.left is None and root.right is None:
                return None

            # 有一个孩子空，则用另一个孩子顶替自己的位置
            elif root.left is None:
                root = root.right
            elif root.right is None:
                root = root.left
            # 都不空
            # 找左子树的最大，只改值，不改指向，简单很多，不用考虑父节点的连接
            else:
                max_node = self.getBSTMax(root.left)   # todo 关键点！
                root.val = max_node.val
                root.left = self.deleteNode(root.left, max_node.val)

        # 注意是BST啊，不是BT，用性质elif分开，别一起考虑，那是BT的做法。
        elif root.val > key:
            root.left = self.deleteNode(root.left, key)
        elif root.val < key:
            root.right = self.deleteNode(root.right, key)
        else:
            pass

        return root

    def getBSTMax(self, root):
        '''
        获取BST的最大值 一定没有右孩子
        :param root:
        :return:
        '''
        while(root.right is not None):
            root = root.right

        return root


# ok
# todo 重点再理解！