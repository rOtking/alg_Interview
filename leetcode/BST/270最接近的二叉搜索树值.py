# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def closestValue1(self, root: Optional[TreeNode], target: float) -> int:
        # 大递归
        if root is None:
            return None
        if root.left is None and root.right is None:
            return root.val

        res1 = self.closestValue(root.left, target)
        res2 = self.closestValue(root.right, target)
        res = root.val
        if res1 is None:
            res = res if abs(res - target) < abs(res2 - target) else res2
        elif res2 is None:
            res = res if abs(res - target) < abs(res1 - target) else res1
        else:
            # 都不空
            res = res if abs(res - target) < abs(res2 - target) else res2
            res = res if abs(res - target) < abs(res1 - target) else res1

        return res

    # BST. 可以递归的中序
    def closestValue(self, root: Optional[TreeNode], target: float) -> int:
        stack = []
        cur = root
        # 记录之前的距离
        pre = -float('inf')

        while (len(stack) != 0 or cur):
            if cur:
                stack.append(cur)
                cur = cur.left
            else:
                cur = stack.pop()
                if pre <= target <= cur.val:
                    break
                pre = cur.val
                cur = cur.right
        if cur:
            return cur.val if abs(cur.val - target) < abs(pre - target) else pre
        else:
            return pre




# todo 核心核心：就是二叉树的迭代中序遍历，记忆记忆！！











