# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:

        if root is None:
            return TreeNode(val)   # todo 注意这是插入！空list也能插入！
        cur = root
        while(cur is not None):
            if cur.val < val:
                if cur.right is not None:
                    cur = cur.right
                else:
                    node = TreeNode(val)
                    cur.right = node
                    break
            else:
                if cur.left is not None:
                    cur = cur.left
                else:
                    node = TreeNode(val)
                    cur.left = node
                    break

        return root

# ok
# 简单