# 介绍，min堆的特性：1.完全BT，依次填满；2.父小于等于两个子
# 插入：1.先放到最后，保证完全BT；2.如果小于等于父，就与父交换，直到满足条件。
# 删除堆顶：1.把最后一个元素移动到堆顶，保证完全BT；2.堆顶如果大于子，就与最小的子交换，直到满足条件。


# todo heapify堆化函数是从下向上移动，从小的堆逐渐合并为大堆；heapInsert是从下往上移动

# todo 解题时，一般不需要自己去实现堆，会用就行了！

# todo 因为是完全BT，可以用arr来表示BT，i位置的父节点是(i - 1)/2，左孩子2i + 1，右孩子2i+2


# 1. 构建堆 时间O(n)，空间O(n) 都要遍历一次
import heapq

# 只有最小堆
# heapify就是堆化，就等于进行了堆排序，调整成堆！从非叶子节点向堆顶，每个位置进行heapify的调整，加起来收敛于O(n)

minHeap = [5,3,6,1]
heapq.heapify(minHeap)
print(minHeap)   # [1, 3, 6, 5]
# 注意，这是BT的排序，不是数组的排序，想得到数组的排序结果，需要依次pop出来！
maxHeap = [5,3,6,1]
maxHeap = [-x for x in maxHeap]
heapq.heapify(maxHeap)
print([-x for x in maxHeap])

# todo 最大堆：乘-1，借助最小堆来实现！

# 2.插入   时间O(logn)   空间O(1)
heapq.heappush(minHeap, 2)

# 大根堆插入    乘-1
heapq.heappush(maxHeap, (-1) * 2)


# 3.获取堆顶   时间O(1)  空间O(1)
# 直接取
minPeek = minHeap[0]
maxPeek = -maxHeap[0]


# 4.pop堆顶  时间O(logn) 空间O(1)
heapq.heappop(minHeap)
print(minHeap)


# 5.获取长度
minLen = len(minHeap)

