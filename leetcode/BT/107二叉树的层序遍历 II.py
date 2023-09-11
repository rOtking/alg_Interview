# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def levelOrderBottom(self, root: TreeNode):
        res = []
        if root is None:
            return res
        lefts = self.levelOrderBottom(root.left)
        rights = self.levelOrderBottom(root.right)

        if len(lefts) == 0 and len(rights) == 0:
            res = [[root.val]]
            return res
        # 有一个空
        if len(lefts) == 0 or len(rights) == 0:
            res = lefts if len(lefts) != 0 else rights
            res.append([root.val])
            return res
        # 都不空
        diff = len(rights) - len(lefts)
        if diff > 0:
        # 右边多，先走右边
            fast = diff
            res.extend(rights[:fast])
        # 此处左右list的起始位置不同了，一个0，一个index
            slow = 0    # 先走几步
            while(fast < len(rights)):
                tmp = lefts[slow]
                tmp.extend(rights[fast])
                res.append(tmp)
                slow += 1
                fast += 1

        elif diff < 0:
            fast = -diff
            res.extend(lefts[:fast])
            slow = 0
            while(fast < len(lefts)):
                tmp = lefts[fast]
                tmp.extend(rights[slow])
                res.append(tmp)
                slow += 1
                fast += 1
        else:
            index = 0
            while(index < len(lefts)):
                tmp = lefts[index]
                tmp.extend(rights[index])
                res.append(tmp)
                index += 1

        res.append([root.val])  # 注意当前节点加入的位置！

        return res

if __name__ == '__main__':
    a = TreeNode(3)
    b = TreeNode(9)
    c = TreeNode(20)
    d = TreeNode(15)
    e = TreeNode(7)


    a.left = b
    a.right = c
    c.left = d
    c.right = e



    s = Solution()
    res = s.levelOrderBottom(a)
    print(res)


# ok
# todo 重点看看！其实不难，细节处理好！先别优化，所有分支都写出来！细节太多了,多走了diff
# todo 可用BFS得到后反转，没什么，不写了。

