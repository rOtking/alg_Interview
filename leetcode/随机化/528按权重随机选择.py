import random
class Solution:

    def __init__(self, w: List[int]):

        self.w = w
        # 构建加权下标范围数组
        self.arr = []
        i, j = 0, w[0] - 1
        self.arr.append([i, j])
        for x in range(1, len(w)):
            i = j + 1
            j = i + w[x] - 1
            self.arr.append([i, j])

        self.total = sum(w)



    def pickIndex(self) -> int:
        index = random.choice([i for i in range(self.total)])
        # todo 超时了！有序的，明显可以用二分来加速！
        # for i in range(len(self.arr)):
        #     if self.arr[i][0] <= index and self.arr[i][1] >= index:
        #         return i
        l, r = 0, len(self.arr) - 1
        while(l <= r):
            mid = l + int((r - l) / 2)
            if self.arr[mid][0] <= index and self.arr[mid][1] >= index:
                return mid
            elif self.arr[mid][0] > index:
                r = mid - 1
            elif self.arr[mid][1] < index:
                l = mid + 1
            else:
                pass


# todo 还没完呢！！ 今天一定要完成啊！！！！


# Your Solution object will be instantiated and called as such:
# obj = Solution(w)
# param_1 = obj.pickIndex()