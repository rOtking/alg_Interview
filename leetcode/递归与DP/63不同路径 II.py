class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid) -> int:
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        dp = [[1] * n for _ in range(m)]
        # base case
        for i in range(m):
            if obstacleGrid[i][0] == 0:
                continue
            else:
                for x in range(i, m):
                    dp[x][0] = 0
                break

        for j in range(n):
            if obstacleGrid[0][j] == 0:
                continue
            else:
                for x in range(j, n):
                    dp[0][x] = 0
                break
        print(dp)

        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j] == 0:
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
                else:
                    dp[i][j] = 0

        return dp[m - 1][n - 1]


if __name__ == '__main__':
    s = Solution()
    res = s.uniquePathsWithObstacles([[1,0]])


# 升级之后套路依然不变
# ok
