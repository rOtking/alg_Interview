# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def sumNumbers(self, root: TreeNode) -> int:
        res = self.help(root)
        r = 0
        for s in res:
            tmp = int(s)
            r += tmp
        return r

    def help(self, root):
        '''
        返回所有子序列[x,...,y]
        x是子序列组成的str
        :param root:
        :return:
        '''
        if root is None:
            return []

        if root.left is None and root.right is None:
            res = [str(root.val)]
            return res

        lefts = self.help(root.left)
        rights = self.help(root.right)

        res = []
        if len(lefts) != 0 and len(rights) != 0:
            res.extend([str(root.val) + i for i in lefts])
            res.extend([str(root.val) + i for i in rights])
        elif len(lefts) != 0:
            res.extend([str(root.val) + i for i in lefts])
        elif len(rights) != 0:
            res.extend([str(root.val) + i for i in rights])
        else:
            pass
        return res
    # DFS. 理解一下！！！！
    # todo BT的DFS就是前序，我说呢。
    def sumNumbersDFS(self, root: TreeNode) -> int:
        # 之前走过的路就算的结果
        def dfs(root, preValue):
            if not root:
                return 0
            preValue = preValue * 10 + root.val
            if not root.left and not root.right:
                return preValue
            return dfs(root.left, preValue) + dfs(root.right, preValue)

        return dfs(root, 0)

    # bfs不直观，就不写了！

if __name__ == '__main__':
    a = TreeNode(0)
    b = TreeNode(1)

    a.left = b

    s = Solution()

    res = s.sumNumbers(a)
    print(res)


# ok
# todo 解法有点low
# todo BFS与DFS的解法
