class Solution:

    # 1. 暴力递归 超时
    def canJump1(self, nums: List[int]) -> bool:

        def dp(i):
            # 能否到达i位置
            if i == 0:
                return True
            for j in range(i):
                if dp(j) and nums[j] >= i - j:
                    return True

            return False

        return dp(len(nums) - 1)

    # 2. memory 超时
    def canJump2(self, nums: List[int]) -> bool:

        def dp(i):
            # 能否到达i位置
            if i == 0:
                return True
            for j in range(i):
                if j not in memo:
                    memo[j] = dp(j)
                if memo[j] and nums[j] >= i - j:
                    return True

            return False

        memo = {}
        return dp(len(nums) - 1)

    # 3. dp 超时
    def canJump3(self, nums: List[int]) -> bool:
        dp = [False] * len(nums)
        dp[0] = True

        for i in range(len(nums)):
            if dp[i]:
                for j in range(i + 1, min(i + nums[i] + 1, len(nums))):
                    dp[j] = True

        return dp[-1]

    # 4. 贪心
    def canJump(self, nums: List[int]) -> bool:
        # 最远能超过nums长度，就可以
        farest = 0

        for i in range(len(nums)):
            if farest >= len(nums) - 1:
                return True
            if farest >= i:
                farest = max(nums[i] + i, farest)
            else:
                break

        return False


# todo 贪心是特殊的dp，能比dp更快！每一步都找局部最优解，合起来就是全局最优解，当然 全局最优=局部最优之和  这个条件并不容易满足！

# todo 思路：记录一个走过的路中能到的最远位置farest，然后在遍历各个位置的过程中更新farest，如果当前位置比farest小，说明当前位置到不了！

# todo 难点：会不会有i位置到不了，i+1位置能到的情况？不存在！！！i能到，之前的位置一定能到！






