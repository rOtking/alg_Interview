class Solution:
    def isInterleave_wrong(self, s1: str, s2: str, s3: str) -> bool:
        if len(s1) + len(s2) != len(s3):
            return False
        i, j, p = 0, 0, 0
        while (i < len(s1) and j < len(s2) and p < len(s3)):
            if s3[p] == s1[i]:
                i += 1
                p += 1
            elif s3[p] == s2[j]:
                j += 1
                p += 1
            else:
                return False

        if p == len(s3):
            return True
        elif i == len(s1):
            if s2[j:] == s3[p:]:
                return True
            else:
                return False

        else:
            if s1[i:] == s3[p:]:
                return True
            else:
                return False
        # todo 这个是错误的，是反例！用指针解决有个致命问题  就是s1与s2的当前位置有s3当前一样是，应该归属哪个呢？会导致问题！

    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        # dp[i][j]为s1[...i]与s2[...j]是否是s3[...i+j]的交错？
        # 取决于 dp[i - 1][j]与dp[i][j - 1]，前面不是，当前一定不是！
        if len(s1) + len(s2) != len(s3):
            return False

        # 注意0位置的''
        dp = [[False] * (len(s2) + 1) for _ in range(len(s1) + 1)]

        # base case
        dp[0][0] = True
        for j in range(1, len(s2) + 1):
            if s2[j - 1] == s3[j - 1]:
                dp[0][j] = True
            else:
                break

        for i in range(1, len(s1) + 1):
            if s1[i - 1] == s3[i - 1]:
                dp[i][0] = True
            else:
                break

        for i in range(1, len(s1) + 1):
            for j in range(1, len(s2) + 1):
                if dp[i - 1][j] and s1[i - 1] == s3[i + j -1]:
                    dp[i][j] = True
                elif dp[i][j - 1] and s2[j - 1] == s3[i + j - 1]:
                    dp[i][j] = True
                else:
                    dp[i][j] = False

        return dp[len(s1)][len(s2)]




# ok
# todo：提示后做出来，要反复看！核心是：dp的定义：有点感觉了，以i与以j为结尾的xxx；还有就是空字符串与index的+-1关系搞清楚！加油！
