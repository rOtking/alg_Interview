# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if root is None:
            return None

        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)

        # 只可能在左右其中之一，或者都没有；不可能两边都出现
        if left is not None:
            return left
        if right is not None:
            return right
        # 两边都没有
        if self.isSonNode(root, p) and self.isSonNode(root, q):
            return root
        else:
            return None
    def isSonNode(self, root, node):
        '''
                root是否是node的祖先
        :param root:
        :param node:
        :return:
        '''
        if root is None:
            return False
        if root is node:
            return True
        else:
            left_flag = self.isSonNode(root.left, node)
            right_flag = self.isSonNode(root.right, node)

            return left_flag or right_flag

# ok
# 大递归 + 判断后代的函数
# todo 重新看！