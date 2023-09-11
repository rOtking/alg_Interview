class Solution:
    def longestPalindrome(self, s: str) -> str:
        # dp[i][j]是i到j是是否回文
        # dp[i][j] = Fales  if s[i] != s[j]
        #    if s[i] == s[j]:   等于dp[i+1][j-1]

        dp = [[False] * len(s) for _ in range(len(s))]

        # base case 对角线与右上一层
        max_distance = 1
        max_i, max_j = 0, 0
        for i in range(len(s)):
            dp[i][i] = True
            j = i + 1
            if j < len(s) and s[i] == s[j]:
                dp[i][j] = True
                max_i, max_j = i, j

        step = 2

        while(True):
            i = 0
            j = i + step
            if j > len(s) - 1:
                break
            else:
                while(j <= len(s) - 1):
                    if s[i] != s[j]:
                        dp[i][j] = False
                    else:
                        dp[i][j] = dp[i + 1][j - 1]
                        if  dp[i][j] and (j - i + 1) > max_distance:
                            max_distance = j - i + 1
                            max_i, max_j = i, j
                    i += 1
                    j += 1
            step += 1

        return s[max_i: max_j + 1] if max_j + 1 <= len(s) - 1 else s[max_i:]

    # 中心扩散   abba  #a#b#b#a#
    def longestPalindrome1(self, s: str) -> str:
        def isValid(s, i, j):
            # 判断s[i:j+1]是不是有效的回文
            if i == j:
                return True

            while(i <= j):
                if s[i] != s[j]:
                    return False

            return True
        def preDeal(s):
            res = '#'
            for i in range(len(s)):
                res += (s[i] + '#')

            return res
        if s is None or len(s) == 0 or len(s) == 1:
            return s
        # 预处理
        chs = preDeal(s)
        pass
    # todo !!!



# ok 完全自己做出来了，dp比之前的做法都简单了不少，当然时间还能优化，但是思想对了！
# todo dp[i][j]表示s[i...j]是否回文
# 当然还可以 遍历每个位置为中心，向两边扩散寻找最长回文，但是重复计算太多了。