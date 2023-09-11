import pprint

class Solution:
    def isSubsequence1(self, s: str, t: str) -> bool:
        # todo 这个dp用的不对，没有降低时间，是dp的定义不对。有更好的dp定义方式，但是双指针才是这题的核心思路，dp不太直观！
        if s == '':
            return True
        if t == '':
            return False

        # dp[i][j] s[...i]是否为t[...j]的子序列
        dp = [[False] * len(t) for _ in range(len(s))]
        for j in range(len(t)):
            dp[0][j] = True if s[0] in t[:j + 1] else False

        for i in range(1, len(s)):
            dp[i][0] = False


        for i in range(1, len(s)):
            for j in range(1, len(t)):
                if dp[i][j - 1]:
                    dp[i][j] = True
                else:
                    if not dp[i - 1][j - 1]:
                        dp[i][j] = False
                    else:
                        dp[i][j] = True if s[i] == t[j] else False

        return dp[len(s) - 1][len(t) - 1]

    def isSubsequence(self, s: str, t: str) -> bool:
        # i在s上，j在t上
        i, j = 0, 0
        while(j < len(t) and i < len(s)):
            if s[i] == t[j]:
                i += 1
                j += 1
            else:
                j += 1
        if i == len(s):
            return True
        else:
            return False


# ok todo 双指针时间山好多了！O(max(m, n))


if __name__ == '__main__':
    sol = Solution()

    s = "axc"

    t = "ahbgdc"
    res = sol.isSubsequence(s, t)
    print(res)