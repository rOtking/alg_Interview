import heapq

class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.arr = []
        heapq.heapify(self.arr)
        for ele in nums:
            if len(self.arr) < self.k:
                heapq.heappush(self.arr, ele)
            elif ele > self.arr[0]:
                self.arr[0] = ele
                heapq.heapify(self.arr)
            else:
                continue

    def add(self, val: int) -> int:
        if len(self.arr) < self.k:
            heapq.heappush(self.arr, val)
        elif val > self.arr[0]:
            self.arr[0] = val
            heapq.heapify(self.arr)
        else:
            pass

        return self.arr[0]



# Your KthLargest object will be instantiated and called as such:
# obj = KthLargest(k, nums)
# param_1 = obj.add(val)


# ok  easy