class Solution:

    # 法1：暴力递归
    def minDistance1(self, word1: str, word2: str) -> int:
        def dp1(i, j):
            # base case
            if i == -1:
                # word1处理完了，如果还不一样，那word还剩多少，word1直接增加这么多就是最小次数
                return j + 1
            if j == -1:
                return i + 1

            if word1[i] == word2[j]:
                return dp1(i - 1, j - 1)
            else:
                return min(dp1(i - 1, j) + 1, dp1(i, j - 1) + 1, dp1(i - 1, j - 1) + 1)

        return dp1(len(word1) - 1, len(word2) - 1)

    # 法2：备忘录
    def minDistance2(self, word1: str, word2: str) -> int:
        def dp2(i, j):
            # base case
            if i == -1:
                # word1处理完了，如果还不一样，那word还剩多少，word1直接增加这么多就是最小次数
                memo[(i, j)] = j + 1
                return memo[(i, j)]
            if j == -1:
                memo[(i, j)] = i + 1
                return memo[(i, j)]

            if (i, j) in memo:
                return memo[(i, j)]

            if word1[i] == word2[j]:
                memo[(i, j)] = dp2(i - 1, j - 1)
                return memo[(i, j)]
            else:
                if (i - 1, j) not in memo:
                    memo[(i - 1, j)] = dp2(i - 1, j)
                if (i, j - 1) not in memo:
                    memo[(i, j - 1)] = dp2(i, j - 1)
                if (i - 1, j - 1) not in memo:
                    memo[(i - 1, j - 1)] = dp2(i - 1, j - 1)
                return min(memo[(i - 1, j)] + 1, memo[(i, j - 1)] + 1, memo[(i - 1, j - 1)] + 1)

        i, j = len(word1) - 1, len(word2) - 1
        memo = {}

        return dp2(i, j)

    # 法3：DP动态规划他来了！！！哈哈
    def minDistance(self, word1: str, word2: str) -> int:
        # 1. 初始化构建dp矩阵 横word1，竖word2，全0
        dp = [[0] * (len(word1) + 1) for _ in range(len(word2) + 1)]

        # 2. 初始条件
        for i in range(len(word1) + 1):
            dp[0][i] = i
        for j in range(len(word2) + 1):
            dp[j][0] = j

        # 3. 由左上角的三个元素推算当前
        for j in range(1, len(word2) + 1):
            for i in range(1, len(word1) + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[j][i] = dp[j - 1][i - 1]
                else:
                    dp[j][i] = min(dp[j][i - 1], dp[j - 1][i - 1], dp[j - 1][i]) + 1

        return dp[len(word2)][len(word1)]








# 思路1：先写暴力递归（状态转移方程）：核心就是划分子问题，原问题dp(i,j)，ij是两个长度，
# dp(i,j) = dp(i-1, j-1)  if word1[i] == word2[j]
#         = min(dp(i-1, j) + 1, dp(i-1, j-1) + 1, dp(i, j-1) + 1)
#          对应word1    删            换            增               3种操作

# 递归下去，显然有大量的重叠子问题

# 思路2：备忘录记下来：搞个数组，把计算过的状态都记下来，每次都先查一下！
# todo 数组备忘录有的时候-1会引发奇怪的问题，那就用dict！

# 思路3：根据递归，画出dp表，找状态方程的规律怎么在dp表上计算。ij位置只与i-1,j、i,j-1、i-1,j-1三个位置有关！
# todo 注意word为""的情况也就是dp[:][0]与dp[0][:]，注意下标与长度即可！

# todo 经典！反复查看！关键：由递推树->状态方程->dp表

if __name__ == '__main__':
    s = Solution()
    res = s.minDistance("horse", "ros")
    print(res)