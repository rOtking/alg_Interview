# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def trimBST(self, root: TreeNode, low: int, high: int) -> TreeNode:
        '''
        当前node小于范围，左子树删了，右子树代替
        大于范围同理
        :param root:
        :param low:
        :param high:
        :return:
        '''
        if root is None:
            return None

        if root.val < low:
            # todo 这是关键啊！不能单纯的子树继承，因为子树也可能也不满足；所以先修建子树，在继承！
            root.right = self.trimBST(root.right, low, high)
            root = root.right

        elif root.val > high:
            root.left = self.trimBST(root.left, low, high)
            root = root.left
        else:
            root.left = self.trimBST(root.left, low, high)
            root.right = self.trimBST(root.right, low, high)

        return root



# ok
# todo 简单，但是需要再看一下

