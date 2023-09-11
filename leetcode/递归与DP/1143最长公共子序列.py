class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        if len(text1) == 0 or len(text2) == 0:
            return 0

        # base case
        dp = [[0] * len(text2) for _ in range(len(text1))]

        for i in range(len(text1)):
            if text2[0] in text1[:i+1]:
                dp[i][0] = 1

        for j in range(len(text2)):
            if text1[0] in text2[:j+1]:
                dp[0][j] = 1

        for i in range(1, len(text1)):
            for j in range(1, len(text2)):
                if text1[i] == text2[j]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[len(text1) - 1][len(text2) - 1]









# todo dp[i][j]是t1[...i]与t2[...j]的最长公共子序列，这个想到了，但是转移方程费了点劲，没想出来！
# dp问题的转移通常比较简单，不要自己搞复杂了！
# 总想着t1[i]与t2[j]不等时，在之前有没有包涵！太复杂了，dp前面的数就是确切答案，即使包含，也在之前解决了！
# todo 如t1[i]是不是在t2[...j]中已经作为子序列了？dp[i][j-1]就已经解决了！

# todo 反复思考哦！ ok了