import heapq

class Solution:
    # 1. 库函数 排序
    def findKthLargest1(self, nums, k: int) -> int:
        nums = sorted(nums, reverse=True)
        return  nums[k - 1]

    # 2. 库函数 堆
    def findKthLargest2(self, nums, k: int) -> int:
        topK = heapq.nlargest(k, nums)
        return topK[-1]

    # 3. 库函数 堆 自己实现k
    def findKthLargest3(self, nums, k: int) -> int:
        arr = [-x for x in nums]
        heapq.heapify(arr)
        while(k - 1 > 0):
            heapq.heappop(arr)
            k -= 1

        return heapq.heappop(arr)

    # 4. 自己维护容量k的堆
    def findKthLargest4(self, nums, k: int) -> int:
        arr = []
        heapq.heapify(arr)
        for ele in nums:
            if len(arr) < k:
                heapq.heappush(arr,ele)
            elif ele > arr[0]:
                arr[0] = ele
                heapq.heapify(arr)
            else:
                continue

        return arr[0]

    # 5. partition
    def findKthLargest(self, nums, k: int) -> int:
        pass

    # 6. 选择排序：部分过程
    def findKthLargest6(self, nums, k: int) -> int:
        for i in range(k):
            max_index = i
            for j in range(i, len(nums)):
                if nums[j] > nums[max_index]:
                    max_index = j

            nums[i], nums[max_index] = nums[max_index], nums[i]

        return nums[k - 1]







