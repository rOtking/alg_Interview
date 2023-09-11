class Solution:
    # 暴力递归，肯定超时！
    # todo 开始优化 哈哈
    def jump1(self, nums) -> int:
        def dp(i):
            # base case
            if i == 0:
                return 0

            # 跳到i位置的最小次数
            min_times = float('inf')
            for j in range(i):
                if nums[j] >= i - j:
                    min_times = dp(j) + 1 if dp(j) + 1 < min_times else min_times

            return min_times

        return dp(len(nums) - 1)

    # 备忘录 还是有点慢
    def jump2(self, nums) -> int:

        def dp(i):
            # base case
            if i == 0:
                return 0

            # 跳到i位置的最小次数
            min_times = float('inf')
            for j in range(i):
                if nums[j] >= i - j:
                    if j not in memory:
                        memory[j] = dp(j)

                    min_times = memory[j] + 1 if memory[j] + 1 < min_times else min_times
            return min_times

        memory = {}

        return dp(len(nums) - 1)

    # dp 还是慢
    def jump3(self, nums) -> int:
        # dp[i]表示从0位置到i 位置最少次数
        dp = [float('inf')] * len(nums)
        dp[0] = 0

        # base case
        for i in range(len(nums) - 1):
            if nums[i] > 0:
                for j in range(i, min(i + nums[i] + 1, len(nums))):
                    dp[j] = min(dp[i] + 1, dp[j])

        return int(dp[len(nums) - 1])


    # 贪心
    # todo 是个区间问题！每次跳的一步都是一个范围，都是一步完成的！
    # 不用遍历最后一个位置，因为已经结束了

    def jump(self, nums) -> int:
        step = 0   # 步数
        farest = 0  # step步能走的最远位置
        end = 0     # 当前step能到的最右边界

        for i in range(len(nums) - 1):

            farest = max(farest, nums[i] + i)

            if i == end:
                step += 1
                end = farest
                # todo 是不是提前结束都无所谓，因为是在一个step中！复杂度不变
                if end >= len(nums) - 1:
                    break
        return step


# todo ok了，费了好大劲，一个反复复习！！！啊啊啊



# todo 核心是：是不是取当前能走的最远值就能保证下次也能走的远？1.走的步数是0-nums[i]全覆盖，不会说中间没走到；2.只要保证当前的选择是能走到的是最远的，
# todo 那也一定包含了其他的选择，所以不存在当前走的近，后面走的远，因为现在走的远的，后面也一定能走这个走的远的道路！
# todo i位置可以x步到达，那0～i-1位置一定能在x步及之内到达！










if __name__ == '__main__':
    s = Solution()
    res = s.jump([2,3,1,1,4])
    print(res)