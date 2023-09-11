# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def searchBST1(self, root: TreeNode, val: int) -> TreeNode:
        '''
        这是BT的做法，不是BST的
        :param root:
        :param val:
        :return:
        '''
        if root is None:
            return None

        if root.val == val:
            return root

        left = self.searchBST1(root.left, val)
        right = self.searchBST1(root.right, val)

        return left or right


    def searchBST(self, root: TreeNode, val: int) -> TreeNode:
        '''
        BST，迭代的方法
        :param root:
        :param val:
        :return:
        '''
        if root is None:
            return None

        cur = root
        res = None
        while(cur):
            if cur.val == val:
                res = cur
                break
            if cur.val < val:
                cur = cur.right
            else:
                cur = cur.left

        return res


# ok
# 没问题！