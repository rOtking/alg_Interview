# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def flatten(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        # 先序遍历形成list，返回tail
        def process(root):
            if root is None or (root.left is None and root.right is None):
                return root

            leftRoot = root.left
            rightRoot = root.right
            root.left = None
            leftTail = process(leftRoot)
            rightTail = process(rightRoot)

            if leftRoot:
                root.right = leftRoot
                leftTail.right = rightRoot
            else:
                root.right = rightRoot
            # todo 尾结点不能是None，不然返回后不能连接
            return rightTail if rightTail else leftTail

        tail = process(root)
        return root
# ok
# 注意，left要断开的！
