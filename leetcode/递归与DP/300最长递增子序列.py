class Solution:
    # 法1：暴力递归
    def lengthOfLIS1(self, nums):
        def dp1(i):
            # 求以第i个位置为结尾的子序列，最大长度是多少
            # todo dp的含义其实才是最核心的

            # base case
            if i == 0:
                return 1

            res = []
            for j in range(i):
                if nums[j] < nums[i]:
                    res.append(dp1(j) + 1)

            return max(res) if len(res) > 0 else 1

        l = 0
        for i in range(len(nums)):
            l = dp1(i) if dp1(i) > l else l

        return l

    # 法2：memory
    def lengthOfLIS2(self, nums):
        def dp2(i):
            # 求以第i个位置为结尾的子序列，最大长度是多少
            # todo dp的含义其实才是最核心的

            # base case
            if i == 0:
                return 1

            res = []
            for j in range(i):
                if nums[j] < nums[i]:
                    if j not in memo:
                        memo[j] = dp2(j)
                    res.append(memo[j] + 1)

            return max(res) if len(res) > 0 else 1

        l = 0
        memo = {}
        for i in range(len(nums)):
            memo[i] = dp2(i)
            l = memo[i] if memo[i] > l else l

        return l

    # 法3:DP
    def lengthOfLIS(self, nums):
        dp = [1 for _ in range(len(nums))]

        for i in range(1, len(nums)):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = dp[j] + 1 if dp[j] + 1 > dp[i] else dp[i]

        return max(dp)
# ok了！
# todo wocao！这个代码也太漂亮了！！！哈哈 精彩！


# todo 其实暴力递归的函数，也就是dp的定义是最难最核心的！特点是：它要有递推关系！
# todo 通常子序列问题：dp[i]是以i位置为结尾的...


if __name__ == '__main__':
    s = Solution()
    r = s.lengthOfLIS([10,9,2,5,3,7,101,18])
    print(r)