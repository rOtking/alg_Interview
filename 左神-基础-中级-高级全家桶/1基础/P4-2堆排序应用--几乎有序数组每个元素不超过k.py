# 对一个几乎有序的数组，几乎有序：每个元素排到正确的位置移动的次数不超过k；k相对n来来说很小。设计排序
'''
todo 方法：搞个k+1容量的小根堆，也就是最小的数一定在0 - k 的范围上；
    1.把0-K上的元素丢入堆，取出堆顶就是全局最小；
    2.去掉0位置，后移，加入K+1位置，重新调整堆
    3.类似滑动窗口，直到结束

todo 可用系统自带的堆

todo 复杂度：每个元素调整都是O(logk),共n个元素，o(nlogk)
'''

import heapq

def sortArrDistanceLessK(arr, k):
    heap = []
    # 因为排序的两个阶段都需要确定一个位置，统一用于1个变量记录更方便
    index1 = 0     # add的遍历位置
    index2 = 0     # pop的排序位置
    # 1.构建k + 1 容量的小根堆
    while(index1 < k + 1):
        heapq.heappush(heap, arr[index1])
        index1 += 1

    # 2.滑动窗口
    while(index1 < len(arr)):
        arr[index2] = heapq.heappop(heap)
        heapq.heappush(heap, arr[index1])
        index2 += 1
        index1 += 1

    # 3.没有push，heap剩余的数字依次pop即可
    while(len(heap) != 0):
        arr[index2] = heapq.heappop(heap)
        index2 += 1

    print(arr)
    return



if __name__ == '__main__':
    arr = [3,2,1,4]
    sortArrDistanceLessK(arr, 3)
