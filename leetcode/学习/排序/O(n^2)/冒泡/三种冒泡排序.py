class Solution:
    # 1. 基础的：一边比较一边换，每轮从待排序区确定一个最大（小）的，放在最后排好。
    def bubbleSort1(self, arr):
        # i就是控制结束
        for i in range(len(arr)):
            for j in range(len(arr) - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]

        return arr

    # todo 其实在每轮遍历过程中，除了最后一个最大数，中间的某些局部位置也经过交换，变得相对有序了！

    # todo 所以，如果上一轮没有经过一次交换就把最后一个位置确定了，那说明待排序的部分已经有序了，就可以停止了！
    # 最好情况，时间O(n)，最差还是O(n^2)

    # 2. 增加一个上一轮是否交换过的记录  ok
    def bubbleSort2(self, arr):
        isSwap = True
        for i in range(len(arr)):
            if not isSwap:
                break
            isSwap = False
            for j in range(len(arr) - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    isSwap = True

        return arr

    # 3. 进一步优化
    # todo 每轮记录一下最后一次交换的位置，下一轮就遍历到上次的最后交换位置即可，因为上一轮没交换的部分肯定有序了，不用再看了！哈哈哈
    def bubbleSort(self, arr):
        isSwap = True
        stopIndex = len(arr) - 1    # 终止位置

        while(stopIndex >= 0):
            if not isSwap:
                break
            isSwap = False
            # 这里实现stopIndex也能正常的-1
            for j in range(stopIndex):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    isSwap = True
                    stopIndex = j

        return arr






if __name__ == '__main__':
    arr = [5,4,3,2,1,9]
    s = Solution()
    res = s.bubbleSort(arr)
    print(res)






