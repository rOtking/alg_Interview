class Solution:
    def __init__(self) -> None:
        self.memo = {}

    def numTrees1(self, n: int) -> int:
        nums = [i for i in range(1, n + 1)]
        return self.process1(nums)

    # 返回nums能够建BST的数量
    def process1(self, nums):

        if len(nums) == 1:
            return 1

        res = 0
        for i, num in enumerate(nums):
            leftNum = self.process1(nums[:i]) if i != 0 else 1
            rightNum = self.process1(nums[i + 1:]) if i != len(nums) - 1 else 1
            res += (leftNum * rightNum)
        return res

    # 复杂度太高不通过---->动态规划！  范围上尝试
    # 构建LR上形成BST的数量. memo的方法
    def process(self, nums, L, R):
        if (L, R) in self.memo:
            return self.memo[(L, R)]
        if L >= R:
            self.memo[(L, R)] = 1
            return self.memo[(L, R)]
        # 至少两个
        res = 0
        for i in range(L, R + 1):
            if (L, i - 1) not in self.memo:
                self.memo[(L, i - 1)] = self.process(nums, L, i - 1)
            if (i + 1, R) not in self.memo:
                self.memo[(i + 1, R)] = self.process(nums, i + 1, R)
            res += (self.memo[(L, i - 1)] * self.memo[(i + 1, R)])
        self.memo[(L, R)] = res
        return res

    def numTrees2(self, n: int) -> int:
        nums = [i for i in range(1, n + 1)]
        return self.process(nums, 0, len(nums) - 1)

    # DP

    # 1 参数两个 LR，0-N 正方形，左下无效. dp[L][R]
    # 2 目标 dp[0][N]. 右上角
    # 3 初始 对角线全1
    # 4 普遍位置 dp[i][j] = dp[i][i-1]*dp[i+1][j] + dp[i][i] * dp[i+2][j] + dp[i][i+1]*dp[i+3][j] + ...+dp[i][j-1]*dp[j+1][j]
    #                                   i     j
    #                             i-1 . . . . . .
    #                             i   * * * . ? .
    #                                 . . . . * .
    #                                 . . . . * .
    #                              j  . . . . * .
    # 5 沿对角线向右上方 或从下向上从左到右 好实现一点
    # todo 边界好复杂！不过是范围枚举DP的典型代表了！！
    def numTrees(self, n: int) -> int:
        dp = [[0] * (n + 2) for _ in range(n + 2)]
        for i in range(n + 2):
            dp[i][i] = 1
            if i + 1 <= n + 1:
                dp[i + 1][i] = 1
        # i是每条斜线起点的横坐标
        for i in range(n - 1, 0, -1):
            for j in range(i + 1, n + 1):
                for p in range(i - 1, j):
                    dp[i][j] += (dp[i][p] * dp[p + 2][j])

        return dp[1][n]
    # 重点完全是边界！！
    # dp[i][j] += (dp[i][p] * dp[q][j])
    # p [i+1, j+1].     q[i-1,j-1]。对应变化的

sol = Solution()
re = sol.numTrees(3)
print(re)





