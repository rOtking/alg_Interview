class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        # dp[i] i为结尾的最大子数组和
        dp = [0] * len(nums)
        dp[0] = nums[0]

        max_sum = dp[0]
        for i in range(1, len(nums)):
            if dp[i - 1] > 0:
                dp[i] = dp[i - 1] + nums[i]
            else:
                dp[i] = nums[i]

            max_sum = max_sum if max_sum > dp[i] else dp[i]

        return max_sum



# ok
# todo 开始还搞dp[i][j]，有点傻了，那不就是暴力搜索么，O(n^2)。
# todo 以i为结尾的dp[i]就是O(n)

# todo 因为有负数，所以ij滑窗不好操作！