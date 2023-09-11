# ------------------ 1. 小和问题 ------------------- #

'''
小和问题:
    数组每个元素左边比它小的数累加。
[1,3,4,2,5]
1左没数字----0；
3左1比它小----1；
4：----- 1 + 3；
2：----- 1
5： 1+3+4+2
'''
'''
正常计算是O(N^2)，利用归并排序时即可O(NlogN)的计算小和
merge的时候 左<右就累加左（右端点-右侧当前位置）次，遍历左侧所有数。
每次merge都是算出当前有序部分的小和，再到更大的部分去计算。
再merge时利用左右相对有序，加速求取过程。

注意这个过程排序是不能省略的！因为排序后才能用下标进行加速！

'''
def smallSum(arr):
    res = process(arr, 0, len(arr) - 1)
    return res

# 左侧自身小和 + 右侧自身 + 两部分构成的小和
def process(arr, left, right):
    if left == right:
        return 0
    mid = left + ((right - left) >> 1)
    l_sum = process(arr, left, mid)
    r_sum = process(arr, mid + 1, right)
    merge_sum = merge(arr,left, mid, right)

    return l_sum + r_sum + merge_sum

def merge(arr, left, mid ,right):
    help_arr = []
    p1 = left
    p2 = mid + 1
    res = 0
    while(p1 <= mid and p2 <= right):
        # todo 很关键。左右相等时，先走右面，且不产生小和。因为这样才能确定右面有多少是真正大的！不然会有相等的情况混淆！
        if arr[p1] < arr[p2]:
            help_arr.append(arr[p1])
            res += arr[p1] * (right - p2 + 1)
            p1 += 1
        else:
            help_arr.append(arr[p2])
            p2 += 1
    # 剩下的左侧都比右边大
    while(p1 <= mid):
        help_arr.append(arr[p1])
        p1 += 1
    # 剩下的右侧都比左边大,和在前面已经算过了！
    while(p2 <= right):
        help_arr.append(arr[p2])
        p2 += 1
    for i in range(left, right + 1):
        arr[i] = help_arr[i - left]

    return res

# 例子
# arr = [1,3,4,2,5]
# res = smallSum(arr)
# print(res)    # ok    得16！

# ------------------------------------------------ #

# ------------------ 2.逆序对问题 ------------------ #
'''
数组在，左面比右面大，就构成一个逆序对。求多少逆序对。
[3,2,4,5,0]
3,2   3,0   2,0   4,0   5,0

即：求每个数左面有多少比他大的数、或右面有多少比他小的数。

与上面的问题一样，就是从大到小排序，在merge的过程中利用下标求左>右的个数。

不写了


总结：左部分内部的结果 + 右部分内部的结果 + 左右之间产生的结果
    左右之间的结果在merge的过程中完成！ 
    利用部分有序下标进行加速。
'''

# ------------------------------------------------ #
'''
讨论数组每个位置左边右边关系的，归并排序考虑一下

'''