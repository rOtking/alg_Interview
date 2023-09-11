# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def __init__(self):
        self.helps = []
    def increasingBST(self, root: TreeNode) -> TreeNode:
        if root is None:
            return None
        self.inOrder(root)
        dummy_root = TreeNode(0)
        cur = dummy_root
        for val in self.helps:
            node = TreeNode(val)
            cur.right = node
            cur = node

        new_root = dummy_root.right
        dummy_root.right = None

        return new_root

    def inOrder(self, root):
        if root is None:
            return

        self.inOrder(root.left)
        self.helps.append(root.val)
        self.inOrder(root.right)

        return


# ok
# easy

