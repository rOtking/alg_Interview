# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def bstFromPreorder(self, preorder: List[int]) -> TreeNode:

        if preorder is None or len(preorder) == 0:
            return None
        root_val = preorder[0]
        root = TreeNode(root_val)

        # 左子
        left = []
        right = []
        for i in preorder[1:]:
            if i < root_val:
                left.append(i)
            else:
                right.append(i)

        root.left = self.bstFromPreorder(left)
        root.right = self.bstFromPreorder(right)

        return root

# ok
