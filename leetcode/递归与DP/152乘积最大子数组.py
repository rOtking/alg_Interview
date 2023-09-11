class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        # dp[i]以i结尾的最大乘与最小乘，因为涉及负负为正
        dp_max = [0] * len(nums)
        dp_min = [0] * len(nums)
        dp_max[0] = nums[0]
        dp_min[0] = nums[0]

        for i in range(1, len(nums)):
            dp_max[i] = max(nums[i], nums[i] * dp_max[i - 1], nums[i] * dp_min[i - 1])
            dp_min[i] = min(nums[i], nums[i] * dp_max[i - 1], nums[i] * dp_min[i - 1])

        return max(dp_max)



# ok 自己做出来了！

# todo 最大最小两个dp，因为涉及负负为正；这里不需要分情况讨论，直接max与min，因为结果就在这个范围里！哈哈 精彩！