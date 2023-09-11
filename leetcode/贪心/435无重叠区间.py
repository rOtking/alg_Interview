class Solution:
    # 贪心
    def eraseOverlapIntervals(self, intervals) -> int:
        intervals = sorted(intervals, key=lambda x:x[1])
        num = 0
        i = 1
        end = intervals[0][1]      # 当前基准的结束时间
        while(i < len(intervals)):
            if intervals[i][0] < end:
                num += 1
            else:
                end = intervals[i][1]
            i += 1
        return num

    # dp：尝试dp能不能做，虽然比贪心慢
    def eraseOverlapIntervals_dp(self, intervals) -> int:
        # dp[i]为第i为最后区间，包含最多的区间数
        intervals = sorted(intervals, key=lambda x:x[1])
        dp = [1] * len(intervals)

        for i in range(1, len(intervals)):
            for j in range(i):
                if intervals[j][1] <= intervals[i][0]:
                    dp[i] = max(dp[j] + 1, dp[i])

        print(dp)
        max_num = max(dp)
        return len(intervals) - max_num




if __name__ == '__main__':
    s = Solution()
    res = s.eraseOverlapIntervals_dp([[1,2],[2,3],[3,4],[1,3]])
    print(res)



# todo 区间调度问题：典型的贪心，按结束时间排序，一次过滤重叠的。
# todo ok 体会思路
# todo 如果dp都很慢，还是O(n^2)，那一定有贪心的解法！
