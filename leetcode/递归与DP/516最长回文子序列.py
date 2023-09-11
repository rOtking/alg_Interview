class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        # i到j上 所以i一定小于等于j，table的右上半部分有效
        dp = [[0] * len(s) for _ in range(len(s))]

        # base case:i==j时，长度就是1
        for x in range(len(s)):
            dp[x][x] = 1

        distance = 1
        for end in range(len(s) - 1, 0, -1):
            # end：i的结束index+1
            for i in range(end):
                j = i + distance
                if s[i] == s[j]:
                    dp[i][j] = dp[i + 1][j - 1] + 2
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])

            distance += 1

        return dp[0][len(s) - 1]

# ok 反复看！



# todo 技术总结：字符串子序列最值问题，都是dp；核心都是dp[]的含义，记几个核心套路：
# 1.最长递增子序列：dp[i]是以s[i]结尾的最长子序列长度；
# 2.2维dp[i][j]：
#             （1）两个str或数组，如最长公共子序列：子数组s1[0...i]与s2[0...j]上，要求子序列的长度；
#              (2)一个str或数组：如最长回文子序列：在s[i...j]上，要求的子序列的长度。

# todo 套路步骤：1.定义dp含义，最好能自己把table画出来；2.确定所求目标，如是dp[][]右下角还是右上角还是别的；3.确定转移、递推的关系，也就是核心逻辑；
#              4.初始化dp的basecase并确定按照什么方向进行递推。

# 本题就直接上dp的做法吧，都挺明确的了：i==j的对角线为1
# 转移方程：dp[i][j] = dp[i+1][j-1] + 2   if s[i] == s[j]
#                  = max(dp[i+1][j], dp[i][j-1])   else
#   i,j-1   |  i,j
#   i+1,j-1 |  i+1,j

# 从对角线，向右上，斜着遍历

