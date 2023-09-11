class Solution:
    def __init__(self):
        self.res = []

    def generateParenthesis(self, n: int):
        self.dfs(track='', n=n)
        return self.res

    def dfs(self, track, n):
        # 1 终止条件
        if len(track) == 2 * n:
            self.res.append(track)
            return

        # 2 一顿忙活，找出决策节点的选择列表，也就是实现决策规则
        left_num = self.calNum(track, '(')
        right_num = self.calNum(track, ')')

        candidates = []
        if left_num < n:
            candidates.append('(')
        if left_num > right_num and right_num < n:
            candidates.append(')')

        # 3 对每个选择尝试
        for candidate in candidates:
            # 4 dfs开始深度优先的尝试
            track += candidate
            self.dfs(track, n)
            # 5 回到上一个记忆点，等待下一轮尝试，stack！
            track = track[:-1]
        return


    def calNum(self, s, mode):
        # 统计数量
        num = 0
        for i in s:
            if mode == i:
                num += 1
        return num

if __name__ == '__main__':
    s = Solution()
    res = s.generateParenthesis(4)
    print(res)


# ok
# todo 居然做对了！有点侥幸，也有点开始真正理解dfs！