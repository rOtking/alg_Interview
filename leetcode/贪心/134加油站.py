class Solution:
    # 暴力解  通过了，但是慢！
    def canCompleteCircuit1(self, gas, cost) -> int:

        for i in range(len(gas)):
            # 调整数组，方便遍历
            tmp_gas = gas[i:]
            tmp_gas.extend(gas[:i])
            tmp_cost = cost[i:]
            tmp_cost.extend(cost[:i])

            j = 0
            res = 0    # 当前剩余油
            while(j < len(tmp_gas) and res >= 0):
                res += tmp_gas[j]
                res -= tmp_cost[j]
                j += 1

            if j == len(tmp_gas) and res >= 0:
                return i

        return -1

    # 图形法 快了点，不明显
    # todo 让最低点开始，余额就再也不会出现负数了！   不通用，还是贪心
    def canCompleteCircuit2(self, gas, cost) -> int:
        # 余额
        res = [gas[i] - cost[i] for i in range(len(gas))]

        if sum(res) < 0:
            return -1
        # 变化趋势
        trend = []
        for i in range(len(res)):
            trend.append(sum(res[:i+1]))

        min_val = min(trend)
        return trend.index(min_val) + 1 if trend.index(min_val) < len(trend) - 1 else 0
    
    # 贪心
    # todo 核心：在一个循环中更新一个值，使其能续上，不用再加一层循环！
    def canCompleteCircuit(self, gas, cost) -> int:















if __name__ == '__main__':
    s = Solution()
    res = s.canCompleteCircuit(gas=[2,3,4], cost=[3,4,3])
    print(res)