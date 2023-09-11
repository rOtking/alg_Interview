# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def __init__(self):
        self.helps = []

    def kthSmallest(self, root: TreeNode, k: int) -> int:
        self.inOrder(root)
        return self.helps[k - 1]


    def inOrder(self, root):
        if root is None:
            return None

        self.inOrder(root.left)
        self.helps.append(root.val)
        self.inOrder(root.right)

    # todo nice!!!
    # 迭代 中序 BST O(K)，上面的方法是o(N)
    def kthSmallest1(self, root: TreeNode, k: int) -> int:
        stack = [root]
        cur = root
        i = 0
        while (len(stack) != 0 or cur):
            if cur:
                stack.append(cur)
                cur = cur.left
            else:
                cur = stack.pop()
                i += 1
                if i == k:
                    break
                cur = cur.right

        return cur.val


# ok
