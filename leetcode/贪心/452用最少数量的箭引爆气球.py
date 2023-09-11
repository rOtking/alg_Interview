class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        points = sorted(points, key=lambda x:x[1])
        i = 1
        num = 0    # 重叠数
        end = points[0][1]    # 当前基准的结束时间
        while(i < len(points)):
            if points[i][0] <= end:
                num += 1
            else:
                end = points[i][1]
            i += 1

        return len(points) - num



# todo 自己没想出来啊！   依然是：求最多不重叠区间！那就是最少的箭！因为每个气球都要命中，所有重叠区间都能1箭解决！


# todo 还是贪心解决重叠区间问题! 反复看！重叠区间问题！


# ok
