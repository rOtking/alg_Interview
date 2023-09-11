class Solution:
    def __init__(self) -> None:
        self.res = []

    def solveNQueens(self, n: int):
        track = []  # track[i]表示第irow放在了哪个列上
        self.dfs(track, 0, n)
        print(self.res)
        # 生成答案
        res1 = []
        for i in range(len(self.res)):
            print(i)
            tmp = []
            for j in range(len(self.res[i])):
                # 生成一行
                tmp1 = []
                for k in range(n):
                    if k != self.res[i][j]:
                        tmp1.append('.')
                    else:
                        tmp1.append('Q')
                tmp.append(''.join(tmp1))
            res1.append(tmp)

        return res1

    # row是该处理第几行了
    # track记录做过的选择
    def dfs(self, track, row, n):
        # 1.终止条件
        if row == n:
            self.res.append(track[:])
            return
        # 2.选择列表：根据之前row的位置，删除掉当前不可放置的位置
        candidates = []
        removeIndexs = set()
        for i in range(row):
            # 相同列
            removeIndexs.add(track[i])
            # 同斜线
            l = track[i] - (row - i)  # 左斜线
            r = track[i] + (row - i)

            if 0 <= l < n:
                removeIndexs.add(l)
            if 0 <= r < n:
                removeIndexs.add(r)

        for i in range(n):
            if i not in removeIndexs:
                candidates.append(i)
        # print(candidates)

        # 3.尝试每个选择
        for candidate in candidates:
            track.append(candidate)
            self.dfs(track, row + 1, n)
            track.pop()

        return


if __name__ == '__main__':
    sol = Solution()
    res1 = sol.solveNQueens(4)
    print(res1)


# ok
# todo 重复看把，细节很多！生成结果的时候有点恶心！

# todo 位运算加速再说吧，不关键，dfs才是关键！即一定搞清track与candidates，就成功一般了！
