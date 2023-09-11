class Solution:
    def countSubstrings(self, s: str) -> int:
        # dp[i][j]：s[i...j]是否回文;
        # dp[i][j] = True    if dp[i + 1][j - 1]为True且s[i] == s[j]
        #          = False   else
        # 斜向上遍历

        dp = [[False] * len(s) for _ in range(len(s))]

        num = 0
        # base case
        for i in range(len(s)):
            dp[i][i] = True
            num += 1
            if i + 1 < len(s):
                if s[i] == s[i + 1]:
                    dp[i][i + 1] = True
                    num += 1
                else:
                    dp[i][i + 1] = False
        print(dp)

        step = 2
        for end in range(len(s) - 3, -1, -1):
            for i in range(end + 1):     # todo 这是重点！end+1才能取到end！
                j = i + step
                if dp[i + 1][j - 1] and s[i] == s[j]:
                    dp[i][j] = True
                    num += 1
                else:
                    dp[i][j] = False

            step += 1
        print(dp)

        return num


# ok  逐渐找到str的dp的感觉了！加油！


if __name__ == '__main__':
    s = Solution()
    res = s.countSubstrings('aaa')