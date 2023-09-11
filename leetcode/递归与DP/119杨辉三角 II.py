class Solution:
    def getRow(self, rowIndex: int) -> List[int]:
        if rowIndex == 0:
            return [1]
        if rowIndex == 1:
            return [1,1]
        pre = [1, 1]

        for i in range(2, rowIndex + 1):
            cur = [1] * (i + 1)
            for j in range(1, i):
                cur[j] = pre[j - 1] + pre[j]

            pre = cur

        return pre





# todo dp的滚动数组来优化空间就是：dp[i]只与dp[i-1]有关，与dp[...i-2]都没关系，那就不需要dp[]，用一个变量不断的更新就好了！

# ok 搞清下标就没问题！哈哈
