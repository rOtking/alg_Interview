# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def __init__(self):
        self.res = 0
    def sumOfLeftLeaves(self, root: TreeNode) -> int:
        self.help(root)
        return self.res

    def isLeft(self, root, node):
        '''
        判断node是否是root的左孩子
        :param root:
        :param node:
        :return:
        '''





