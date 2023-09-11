# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def __init__(self):
        self.help = []

    def recoverTree1(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        self.inOrder(root)
        indexs = []
        for index in range(len(self.help) - 1):
            if self.help[index] > self.help[index + 1]:
                indexs.append(index)

        # 记录一个或者两个
        exchanges = None
        if len(indexs) == 1:
            exchanges = (self.help[indexs[0]], self.help[indexs[0] + 1])
        else:
            exchanges = (self.help[indexs[0]], self.help[indexs[1] + 1])
        self.exchange(root, exchanges)

    def exchange(self, root, exchanges):
        if root is None:
            return

        if root.val == exchanges[0]:
            root.val = exchanges[1]
        elif root.val == exchanges[1]:
            root.val = exchanges[0]
        self.exchange(root.left, exchanges)
        self.exchange(root.right, exchanges)
        return

    def inOrder(self, root):
        if root is None:
            return

        self.inOrder(root.left)
        self.help.append(root.val)
        self.inOrder(root.right)

    # 注意 看清是交换节点还是换两个值就行，差别很大的！换值很简单的
    def recoverTree2(self, root: TreeNode) -> None:
        stack = []
        res = []
        cur = root
        pre = TreeNode(-float('inf'))
        while (len(stack) != 0 or cur):
            if cur:
                stack.append(cur)
                cur = cur.left
            else:
                cur = stack.pop()
                # todo 不能看左右，得看加入res的前后
                # errorFlag = (True if cur.left and cur.left.val >= cur.val else False) or (True if cur.right and cur.right.val <= cur.val else False)
                # if errorFlag:
                #     res.append(cur)
                if cur.val <= pre.val:
                    res.append((pre, cur))
                pre = cur

                cur = cur.right
        # 交换值
        # 待交换的两个数，x是第一个出现的pre，y是最后一个pre后面的cur，xy可能连接也可能不连接！  很关键！
        x, y = res[0][0], res[-1][-1]
        self.swap((x, y))
        return

    def swap(self, pair):
        tmp = pair[0].val
        pair[0].val = pair[1].val
        pair[1].val = tmp
        # 迭代的方式成功了，但是空间是O(H)，H是BT的深度，想实现O(1)，需要Morris遍历！！！！

    # morris中序
    def recoverTree(self, root: TreeNode) -> None:
        cur = root
        mostRight = None
        pre = TreeNode(-float('inf'))
        res = []
        while (cur):
            mostRight = cur.left
            if mostRight:
                while (mostRight.right and mostRight.right is not cur):
                    mostRight = mostRight.right

                if mostRight.right is None:
                    mostRight.right = cur
                    cur = cur.left
                    continue
                else:
                    mostRight.right = None
            # 中序输出点
            if cur.val <= pre.val:
                res.append((pre, cur))
            pre = cur
            cur = cur.right

        x, y = res[0][0], res[-1][-1]
        self.swap((x, y))

        return

# ok
# todo   写出来了！三种方法，空间逐步降低，但是核心是对Morris的三种遍历方式很熟悉！一定重复看啊啊啊！！！
