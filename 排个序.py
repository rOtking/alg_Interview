
# todo end--；与i与i+1比较，终止在end-1
# 基础版本
def bubbleSort1(nums):
    for end in range(len(nums) - 1, -1, -1):
        for i in range(end):
            # i与i+1比较，所以终止在end-1，正好可以和end进行比较并交换
            if nums[i] > nums[i + 1]:
                nums[i], nums[i + 1] = nums[i + 1], nums[i]
    print(nums)
    return nums

# todo 当前轮没发生过交换，直接返回
def bubbleSort2(nums):
    isSwap = True
    for end in range(len(nums) - 1, -1, -1):
        if not isSwap:
            break
        isSwap = False
        for i in range(end):
            if nums[i] > nums[i + 1]:
                nums[i], nums[i + 1] = nums[i + 1], nums[i]
                isSwap = True
    print(nums)
    return nums

# todo 记录上一轮最后一次交换的位置，因为后面没交换的肯定已经有序了
def bubbleSort3(nums):
    isSwap = True
    lastChange = len(nums) - 1
    while(lastChange >= 0):
        if not isSwap:
            break
        isSwap = False
        newLastChange = 0
        for i in range(lastChange):
            if nums[i] > nums[i + 1]:
                nums[i], nums[i + 1] = nums[i + 1], nums[i]
                isSwap = True
                newLastChange = i
        lastChange = newLastChange
    print(nums)

# todo [0, end]已经排好,end+1 从后向前滑到nums[end+1] > nums[end]的位置停止
# todo i与i-1位置换
def insertSort(nums):
    for end in range(len(nums) - 1):
        for i in range(end + 1, 0, -1):
            if nums[i] < nums[i - 1]:
                nums[i], nums[i - 1] = nums[i - 1], nums[i]
            else:
                break
    print(nums)
    return nums

# todo 当前最小位置minIndex
def selectSort(nums):
    for i in range(len(nums)):
        minIndex = i
        for j in range(i, len(nums)):
            if nums[j] < nums[minIndex]:
                minIndex = j
        nums[i], nums[minIndex] = nums[minIndex], nums[i]
    print(nums)

    return nums


# todo 递归！左面排好+右面排好+merge的过程中完成排序！
def mergeSort(nums):
    mergeSortProcess(nums, 0, len(nums) - 1)
    print(nums)
    return nums

def mergeSortProcess(nums, l, r):
    if l == r:
        return
    mid = l + ((r - l) >> 1)     # todo 优先级 很关键！！！！
    mergeSortProcess(nums, l, mid)
    mergeSortProcess(nums, mid + 1, r)
    merge(nums, l, mid, mid + 1, r)

def merge(nums, left_head, left_tail, right_head, right_tail):
    p1 = left_head
    p2 = right_head
    help = []
    while(p1 <= left_tail and p2 <= right_tail):
        if nums[p1] <= nums[p2]:
            help.append(nums[p1])
            p1 += 1
        else:
            help.append(nums[p2])
            p2 += 1
    while(p1 <= left_tail):
        help.append(nums[p1])
        p1 += 1
    while(p2 <= right_tail):
        help.append(nums[p2])
        p2 += 1
    for i in range(left_head, right_tail + 1):
        nums[i] = help[i - left_head]

import random
def quickSort(nums):
     quickProcess(nums, 0, len(nums) - 1)
     print(nums)
     return nums

def quickProcess(nums, left, right):
    if left >= right:
        return
    tmp = left + int(random.random() * (right - left + 1))
    nums[tmp], nums[right] = nums[right], nums[tmp]
    small_right, big_left = partition(nums, left, right)
    quickProcess(nums, left, small_right)
    quickProcess(nums, big_left, right)
    return

def partition(nums, left, right):
    pivot = nums[right]
    small_right, big_left = left - 1, right
    i = left
    while(i < big_left):
        if nums[i] < pivot:
            small_right += 1
            # todo 这里关键！！！
            nums[small_right], nums[i] = nums[i], nums[small_right ]
            i += 1
        elif nums[i] == pivot:
            i += 1
        else:
            big_left -= 1
            nums[i], nums[big_left] = nums[big_left], nums[i]
    nums[i], nums[right] = nums[right], nums[i]
    big_left += 1
    # 小于区右边界与大于区左边界
    return small_right, big_left


# 堆
# todo 堆的实现与堆排序不一样！
#  堆要实现：add（具体是heapInsert），pop（heapify），peek，size操作------核心是 heapInsert（从下向上），heapify（从上向下）
#  堆排序：（非堆）先堆化--pop顶，尾移到顶--（非堆）堆化---...     堆排序是不断堆化的过程
#  关键是初始化时完全乱序，堆化两种方法：1.从头到尾依次heapInsert；2.从尾到头依次heapify ------ 不管哪种，遍历完就是堆了
#  之后的 非堆其实是除了堆顶 下面都是堆，所以，下面不用依次堆化，只对堆顶堆化即可！ 所以理论上，堆排序只需要实现heapify，并不需要heapInsert

def heapSort(nums):
    # heapify是时间收敛于O(n)
    # for i in range(len(nums) - 1, -1, -1):
    #     heapify(nums, i, len(nums))
    # heapInsert初始化 O(NlogN)
    for i in range(len(nums)):
        heapInsert(nums, i)
    heapSize = len(nums) - 1
    while(heapSize > 0):
        nums[0], nums[heapSize] = nums[heapSize], nums[0]
        heapify(nums, 0, heapSize)
        heapSize -= 1
    print(nums)
    return nums

# heapSize：（向下的过程）越界的终止条件
# todo 从0位置开始多少个元素是堆
def heapify(nums, index, heapSize):
    left = 2 * index + 1
    while(left < heapSize):
        right = left + 1
        largest = right if right < heapSize and nums[right] > nums[left] else left
        largest = index if nums[index] > nums[largest] else largest
        if largest == index:
            break
        nums[index], nums[largest] = nums[largest], nums[index]
        index = largest
        left = index * 2 +1

# 辅助
# todo 插入过程只保证插入位置以上是堆，下面的不保证，所以不需要heapSize
def heapInsert(nums, index):
    parent = int((index - 1) / 2)
    while(nums[index] > nums[parent]):
        nums[index], nums[parent] = nums[parent], nums[index]
        index = parent
        parent = int((index - 1) / 2)


# topK问题：无序arr，求第k大的数，或最大的k个数，一样的。
# todo
#  1.排序后
#  2.冒泡，选择都可以，迭代k轮即可，时间 O(nk)
#  3.python的 heapy 的api：核心还是堆的应用
#  求K个最大，用小根堆，堆顶最小，新来一个数，比堆顶小抛弃（比当前的k个小那它一定进不了前k）；比堆顶大则把堆顶换了，堆化。
#  4.自己实现堆：（1）所有数堆化为大根堆，pop k次即可：占空间 O(n)，时间 = 初始化堆时间O(nlogn) + 取K个时间是O(klogn)
#             （2）大小为k的小根堆，在线比较，更好！空间O(k),时间 = 初始化O(klogk) + 调整O(nlogk)   就是空间优化了！
#  5.partition:每次确定一个位置index，如果是k直接停止了，小-等-大已经确定了！不满足就继续二分的找k即可！时间

import heapq
# python api
def topK1(nums, k):
    heap = nums[:k]
    heapq.heapify(heap)
    for i in range(k + 1, len(nums)):
        if nums[i] > heap[0]:
            heap[0] = nums[i]
            heapq.heapify(heap)
    print(heap)

# 自己维护k大小的堆
def topK2(nums, k):
    # 因为实现的是大根堆，用负数来实现
    help = [-i for i in nums]
    heap = help[:k]
    for i in range(k):
        heapInsert(heap, i)
    for i in range(k + 1, len(help)):
        if help[i] < heap[0]:
            heap[0] = help[i]
            heapify(heap, 0, len(heap))
    print(heap)
    res = [-i for i in heap]
    print(res)

# partition
def topK3(nums, k):
    left, right = 0, len(nums) - 1
    while(left <= right):
        # 小于区右边界与大于区左边界
        smallLeft, bigRight = partition(nums, left, right)
        if smallLeft + 1 <= len(nums) - k <= bigRight - 1:
            print(nums[(len(nums) - k):])
            return nums[(len(nums) - k):]
        elif len(nums) - k < smallLeft + 1:
            right = smallLeft
        elif len(nums) - k > bigRight - 1:
            left = bigRight
        else:
            pass
    return








# 二分
# todo 搜索区间是[left,right] 所以终止条件是 while(left<=right) 如果是left<right，那l==r是结果时就丢了！
# todo 就是搞清 搜索区间 与 终止条件！   while(left<=right)的终止条件是 left == right + 1
def binarySearch(nums, target):
    left, right = 0, len(nums) - 1
    while(left <= right):
        mid = left + ((right - left) >> 1)
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        elif nums[mid] > target:
            right = mid - 1
        else:
            pass
    return -1

# 寻找左边界 [1, 2, 2, 2, 2, 3, 4, 5] 2的左边界是index=1
def binarySearchLeft(nums, target):
    left, right = 0, len(nums) - 1
    while(left <= right):
        mid = left + ((right - left) >> 1)
        if nums[mid] < target:
            left = mid + 1
        elif nums[mid] > target:
            right = mid - 1
        elif nums[mid] == target:
            right = mid - 1
        else:
            pass
    # todo 出来时 left == right + 1，有三种情况：
    #  1.right在index=0的左边，left在0位置,且num[left] != target--->target比所有数都小；
    #  2.right在len(nums) - 1,left超出索引，------->target比所有数都大；
    #  3.left在正常位置且nums[left] == target，那她就是左边界
    #  只要存在就一定是left
    if 0 <= left <= len(nums) - 1 and nums[left] == target:
        return left
    else:
        return -1

# 右边界同理
def binarySearchRight(nums, target):
    left, right = 0, len(nums) - 1
    while(left <= right):
        mid = left + ((right - left) >> 1)
        if nums[mid] < target:
            left = mid + 1
        elif nums[mid] > target:
            right = mid - 1
        elif nums[mid] == target:
            left = mid + 1
        else:
            pass
    if 0 <= right <= len(nums) - 1 and nums[right] == target:
        return right
    else:
        return -1

# todo 超大文件搜索，内存一次放不下：
#  1.离线将所有文件hash--->out很分散
#  2.out % n 所有数均匀分到n个同，且相同文件一定分到一个桶内
#  3.query也 hash 后 %n，如得到m号桶
#  4.把m桶的销量数据载入内存搜索即可


# todo 超大数组排序：归并！先分成n组，组内排好序，组间外排。


# todo 常数时间获取最小值的栈
#     stack =     [2 3 3 1 4]
#  minStack = [inf 2 2 2 1 1]   随时取最小
class MinStack:
    # todo 难点是getMin，因为你记录的min可能pop掉，那pop后min该是多少？
    # todo 维护两个stack，其中一个累积放min值
    def __init__(self):
        self.stack = []
        self.minStack = [float('inf')]

    def push(self, val: int) -> None:
        self.stack.append(val)
        if val < self.minStack[-1]:
            self.minStack.append(val)
        else:
            tmp = self.minStack[-1]
            self.minStack.append(tmp)

    def pop(self) -> None:
        self.stack.pop()
        self.minStack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.minStack[-1]


# 179. 最大数
# todo sorted重构比较函数
# todo cmp_to_key中的函数就是比较规则，你可以默认 初始排序是 x小，y大  cmp是规则
#  1.如果规则返回负数，那排序就是y在前 y大，x小     也就是从大到小：即sorted(reverse=True)的实质
#  2.如果返回正，那就是x在前   x小，y大           也就是从小到大：即sorted的默认
#  3.如果为0，那就是保持不动。
# todo 直接的 sorted(hashmap.items(), key=lambda x:x[1])
#  key就是说x就是可迭代的那个元素，冒号后的：x[1]就是用什么来排序，默认是数值大小，从小到大。想自定义就是cmp_to_key

import functools
class Solution:
    def largestNumber(self, nums) -> str:
        def cmp(str1, str2):
            # todo 不用去尝试完成规则，直接得到结果比较不就好了？穷举也就2个结果啊！哈哈
            #  返回负数，则a在b前，正数则a在b后，0不变。
            res1 = int(str1 + str2)
            res2 = int(str2 + str1)
            if res1 > res2:
                return 1
            elif res1 < res2:
                return -1
            else:
                return 0

        res = ''
        nums_str = [str(x) for x in nums]
        nums_str = sorted(nums_str, key=functools.cmp_to_key(cmp), reverse=True)

        for s in nums_str:
            res += s

        # 处理开头是0
        if res[0] == '0' and len(res) > 0:
            return '0'

        return res

def cmp(x, y):
    if x == 11 and y != 11:
        return -1
    elif x == 11 and y == 11:
        return 0
    else:
        return 1

if __name__ == "__main__":
    a= [4,3,1,2,2,7,2,2,5,5]
    # a = heapSort(a)
    # print('--------')
    # indx = binarySearch(a, 2)
    # print(indx)
    # indx = binarySearchLeft(a, 2)
    # print(indx)
    # indx = binarySearchRight(a, 2)
    # print(indx)

    # topK3(a, 3)

    aa = {'a':99,'b':100,'c':1}
    # 按value逆序存key   100>99>1   则 bac
    b = sorted(aa.items(), key=lambda x:x[1])
    print(b)





