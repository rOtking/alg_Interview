# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def postorderTraversal(self, root: TreeNode):
        res = []
        if not root:
            return res
        # 前序‘是根-右-左，反过来就是左右根的后续
        stack = [root]
        while(len(stack) != 0):
            cur = stack.pop()
            res.append(cur.val)
            if cur.left:
                stack.append(cur.left)
            if cur.right:
                stack.append(cur.right)
        res = res[::-1]
        # 逆序的三种方法：1.res = res[::-1] 2.res.reverse()  3.reversed(res)
        return res

    # 挑战Morris
    def postorderTraversalMorris(self, root: TreeNode):
        res = []
        if not root:
            return res

        cur = root
        mostRight = None
        while (cur):
            mostRight = cur.left
            if mostRight:
                while (mostRight.right and mostRight.right is not cur):
                    mostRight = mostRight.right
                # 指向空 第一次到cur
                if mostRight.right is None:
                    mostRight.right = cur
                    cur = cur.left
                    continue
                else:
                    mostRight.right = None
                    res.extend(self.printReverse(cur.left))

            cur = cur.right
        res.extend(self.printReverse(root))
        return res

    def printReverse(self, head):
        # 逆序打印
        res = []
        if head is None:
            return res

        newHead = self.reverse(head)
        cur = newHead
        while (cur):
            res.append(cur.val)
            cur = cur.right
        _ = self.reverse(newHead)
        return res

    def reverse(self, head):
        # BT按right逆序
        if not head or not head.right:
            return head

        pre = None
        cur = head
        while (cur):
            next_ = cur.right
            cur.right = pre
            pre = cur
            cur = next_
        return pre


if __name__ == '__main__':
    a = TreeNode(1)
    b = TreeNode(2)
    c = TreeNode(3)
    d = TreeNode(4)
    e = TreeNode(5)

    a.left = b
    a.right = c
    b.left = d
    b.right = e

    s = Solution()
    res = s.postorderTraversal(a)
    print(res)


# ok
# todo 重点看！！！