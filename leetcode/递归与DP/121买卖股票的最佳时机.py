class Solution:
    # 超时了
    def maxProfit1(self, prices: List[int]) -> int:
        max_distance = 0
        for i in range(len(prices)):
            for j in range(i + 1, len(prices)):
                max_distance = prices[j] - prices[i] if prices[j] - prices[i] > max_distance else max_distance

        return max_distance

    def maxProfit(self, prices: List[int]) -> int:
        # dp[i]以i卖出的最大，可以滚动数组的思想优化空间
        min_value = prices[0]  # i -1位置及之前的最小值
        max_distance = 0
        for i in range(1, len(prices)):
            max_distance = prices[i] - min_value if prices[i] - min_value > max_distance else max_distance
            min_value = prices[i] if prices[i] < min_value else min_value

        return max_distance


# ok 关键是怎么节省循环？
# todo 核心是以i为结尾（卖出）的dp典型定义方式，将双变的问题变成单变。dp1是以i为结尾的最大，dp2是i之前的最小。接着用滚动数组的方式优化空间即可。
