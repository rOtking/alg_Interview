# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def __init__(self):
        self.helps = []
    def inOrder(self, root):
        if root is None:
            return
        self.inOrder(root.left)
        self.helps.append(root.val)
        self.inOrder(root.right)

    def getMinimumDifference(self, root: TreeNode) -> int:
        '''
        BST的中序，最小一定相邻
        :param root:
        :return:
        '''
        self.inOrder(root)
        diff = float('inf')
        for index in range(len(self.helps) - 1):
            tmp = self.helps[index + 1] - self.helps[index]
            diff = tmp if tmp < diff else diff
        return diff

# ok
# BST的中序特性