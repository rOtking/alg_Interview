class MedianFinder:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.cache = []
        self.i = 0
        self.j = 0

    def addNum(self, num: int) -> None:
        if len(self.cache) == 0:
            self.cache.append(num)
            return

        insert_index = None
        for index, ele in enumerate(self.cache):
            if num < ele:
                insert_index = index
                break
        if insert_index is not None:
            self.cache.insert(insert_index, num)
        else:
            self.cache.append(num)
            insert_index = len(self.cache) - 1

        if self.i == self.j:
            self.j += 1  # 其实是3种情况，但都是j+1，画图最好理解
        else:
            self.i += 1

    def findMedian(self) -> float:
        if self.i == self.j:
            return self.cache[self.i]
        else:
            return (self.cache[self.i] + self.cache[self.j]) / 2


# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()


# todo 这个题偏设计，用到了双指针
# Your MedianFinder object will be instantiated and called as such:

obj = MedianFinder()
obj.addNum(-1)
obj.addNum(-2)
obj.addNum(-3)
res = obj.findMedian()
print(res)

# ok
# todo 但是太好时，可以借助有序数组 from sortedcontainers import SortedList 来提升排序的步骤。