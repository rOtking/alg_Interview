class Solution:
    def numberOfBoomerangs(self, points: List[List[int]]) -> int:
        res = 0
        # 遍历所有拐点，统计其他点到拐点的距离
        for point in points:
            distance2num = {}
            for other in points:
                if other is not point:
                    dis = (point[0] - other[0]) * (point[0] - other[0]) + (point[1] - other[1]) * (point[1] - other[1])
                    if dis in distance2num:
                        distance2num[dis] += 1
                    else:
                        distance2num[dis] = 1
            # m个相等就是Am2，全排列
            for k, v in distance2num.items():
                if v >= 2:
                    res += v * (v - 1)

        return res

# ok 就是遍历，排列组合。
# todo 注意hash的运用，逃不开遍历。