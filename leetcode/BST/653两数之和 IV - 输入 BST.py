# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def __init__(self):
        self.helps = []

    def findTarget(self, root: TreeNode, k: int) -> bool:
        self.inOrder(root)

        start = 0
        end = len(self.helps) - 1

        flag = False
        while(start < end):
            if self.helps[start] + self.helps[end] == k:
                flag = True
                break
            elif self.helps[start] + self.helps[end] < k:
                start += 1
            else:
                end -= 1
        return flag

    def inOrder(self, root):
        if root is None:
            return
        self.inOrder(root.left)
        self.helps.append(root.val)
        self.inOrder(root.right)

        return


# ojbk
# 简单
