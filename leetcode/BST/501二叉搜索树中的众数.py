# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def __init__(self):
        self.tables = {}
    def help(self, root):
        '''
        统计每个数值的出现次数
        :param root:
        :return:
        '''
        if root is None:
            return
        if root.val in self.tables:
            self.tables[root.val] += 1
        else:
            self.tables[root.val] = 1

        self.help(root.left)
        self.help(root.right)

    def findMode1(self, root: TreeNode) -> List[int]:
        '''
        法一：O（n）额外空间统计频次
        :param root:
        :return:
        '''
        self.help(root)
        res = []
        # dict排序
        if len(self.tables) != 0:
            # 得到list！
            self.tables = sorted(self.tables.items(), key=lambda x:x[1], reverse=True)
            key = self.tables[0][1]
            for k, v in self.tables:
                if v == key:
                    res.append(k)
                else:
                    break
        return res

    def findMode(self, root: TreeNode) -> List[int]:
        '''
        BST的中序遍历！一定非递减！！
        也就是求：最长连续序列！
        :param root:
        :return:
        '''
        pass


# ok
# todo 想实现O（1）的空间，要用Morris中序遍历，待定
