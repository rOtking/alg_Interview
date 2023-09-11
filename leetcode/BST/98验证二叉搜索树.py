# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def __init__(self):
        self.res = []

    def isValidBST1(self, root: TreeNode) -> bool:
        # 中序性质：一定是从小到大
        self.process1(root)
        self.res.insert(0, float('-inf'))
        self.res.append(float('inf'))
        flag = True
        for i in range(1, len(self.res) - 1):
            if self.res[i - 1] >= self.res[i] or self.res[i] >= self.res[i + 1]:
                flag = False
                break
        return flag

    def process1(self, root):
        if root is None:
            return

        self.process1(root.left)
        self.res.append(root.val)
        self.process1(root.right)

    def isValidBST2(self, root: TreeNode) -> bool:
        # 迭代 复杂的中序
        stack = []
        cur = root
        res = []
        while (len(stack) != 0 or cur):
            if cur:
                stack.append(cur)
                cur = cur.left
            else:
                cur = stack.pop()
                res.append(cur.val)
                cur = cur.right

        for i in range(len(res) - 1):
            if res[i] >= res[i + 1]:
                return False
        return True

    # 树型DP递归
    def isValidBST3(self, root: TreeNode) -> bool:
        # 返回子树是不是以及最大最小
        def process(root):
            if root is None:
                return None
            if root.left is None and root.right is None:
                return True, root.val, root.val
            res1 = process(root.left)
            res2 = process(root.right)
            if res1 and res2:
                isValid = True if res1[0] and res2[0] and res1[2] < root.val < res2[1] else False
                return isValid, res1[1], res2[2]
            elif res1:
                isValid = True if res1[0] and res1[2] < root.val else False
                return isValid, res1[1], root.val
            else:
                isValid = True if res2[0] and root.val < res2[1] else False
                return isValid, root.val, res2[2]

        return process(root)[0]

    # 这种貌似不能DP
    # def isValidBST(self, root: TreeNode) -> bool:


if __name__ == '__main__':
    a = TreeNode(5)
    b = TreeNode(1)
    c = TreeNode(4)
    d = TreeNode(3)
    e = TreeNode(6)

    a.left = b
    a.right = c

    c.left = d
    c.right = e

    s = Solution()
    res = s.isValidBST(a)
    print(res)


# ok
# todo 多做几次！很重要！