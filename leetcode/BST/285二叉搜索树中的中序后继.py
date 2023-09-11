# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def inorderSuccessor(self, root: 'TreeNode', p: 'TreeNode') -> 'TreeNode':
        # 中序遍历没啥意思 拿来练练迭代吧
        if root is None:
            return root
        stack = []
        cur = root
        pre = None
        while (len(stack) != 0 or cur):
            if cur:
                stack.append(cur)
                cur = cur.left
            else:
                cur = stack.pop()
                if pre is p:
                    break
                pre = cur
                cur = cur.right
        return cur

