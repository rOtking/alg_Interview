# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    # def inorderTraversal(self, root: TreeNode) -> List[int]:
    # res = []
    # if root is None:
    #     return res

    # res.extend(self.inorderTraversal(root.left))
    # res.append(root.val)
    # res.extend(self.inorderTraversal(root.right))

    # return res

    # todo 迭代. 很关键 要记忆!!!
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        if root is None:
            return res
        cur = root
        stack = []
        while (len(stack) != 0 or cur):
            if cur:
                stack.append(cur)
                cur = cur.left
            else:
                cur = stack.pop()
                res.append(cur.val)
                cur = cur.right
        return res



