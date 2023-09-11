# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def upsideDownBinaryTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        # 返回新root
        if root is None or root.left is None:
            return root

        left = root.left
        right = root.right
        leftRoot = self.upsideDownBinaryTree(left)
        rightRoot = self.upsideDownBinaryTree(right)

        root.left = None
        root.right = None

        left.right = root
        left.left = right

        return leftRoot



# 搞清题意就行了   大递归！