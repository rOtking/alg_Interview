# ------------------- 1.引入问题 ------------------#
'''
给定num，在O(n)把arr分为<=num与>num两个区域。两区域内部可以无序，整体划分了即可。

'''

'''
} 5，4，3，6，8，1     num=5


维护一个变量，记录 <= 区域的右边界，初始在arr最左侧；遍历i位置，三种情况：
（1）arr[i] <= num，arr[i]与右边界下一个数交换，右边界扩一个，i也加1；
（2）arr[i] > num，i++

模型：
[小于等于区域 | 大于区域 | i,待定区域]
i是当前位置

就是"小于等于区域"把"大于区域"推着向右走，直到arr结束。
'''
# ----------------------------------------------- #

# --------------------- 2.升级：荷兰国旗问题 -------------------#
'''
[< , == , >]三个区域严格分开，内部可以无序。

} 5，4，3，6，8，1 {

一个记录 < 区域右边界，初始在arr最左；一个记录 > 区域左边界，初始在arr最右。
遍历i，3种情况
（1）arr[i] < num：arr[i]与右边界下一个数交换，右边界扩一个，i++；
（2）arr[i] == num：i++
（3）arr[i] > num：arr[i]与左边界前一个数交换，左边界左扩1，i在原地。因为i新换过来的还没判断，需要继续判断。
i 与 小于区域左边界撞上时停止。


模型：
[小于区域 | 等于区域 | i，待定区域 | 大于区域]
'''
# ----------------------------------------------- #

# ---------------------- 3.快排1.0版本 --------------#
'''
arr的最后一个数做划分值pivot，
[.......,pivot]
（1）利用partition函数将arr分为 [<=pivot, >pivot, pivot]
（2）将pivot与 >pivot区域的第一个元素交换，此时就完成了划分；
实质就是一次partition划分，确定了pivot的正确位置
（3）递归将左右两部分继续partition，每次排好一个位置，最终划分完也就排完了。
'''
# ----------------------------------------------- #
# ---------------------- 4.快排2.0版本 --------------#
'''
就是荷兰国旗的三区域partition，稍快一点。


分析：
1，2，3，4，5，6，7
类似构造的有序数组，每次的pivot都很偏，最差O(N ^ 2)

最好是每次都正好二分数组，O(NlogN)
T(N) = 2 * T(N /2) + O(N)
partition是O(N)

可改进。
'''
# ----------------------------------------------- #

# ---------------------- 5.快排3.0版本 --------------#
'''
partition时，随机选个位置做pivot，再把它与arr结尾交换。

因为随机，所以好坏情况都是概率事件。

不管pivot偏不偏，每种情况都是等概率，都列出来求期望,O(NlogN)

依旧是 二叉树模型
第一次partition确定1个数位置；
第二次，  最多确定2个数位置；
第三次，  最多确定4个数位置，最少确定1个。
...
遍历次数logN~N次，每次O(N)


空间复杂度：即求记录多少个pivot的位置，最差每次打偏，开N层递归，O(N)；最好是开logN层，O(logN)


下面实现
'''
import random

def quickSort(arr):
    process(arr, 0, len(arr) - 1)
    return
# 完成l-r上的排序
def process(arr, left, right):
    # todo 区域至少两个数字才能排序，0或1个直接返回
    if left >= right:
        return
    # [0,1)
    tmp = left + int(random.random() * (right - left + 1))
    arr[tmp], arr[right] = arr[right], arr[tmp]
    sr, bl = partition(arr, left, right)  # sr是小于区域右边界，bl是大于区域左边界
    process(arr, left, sr - 1)
    process(arr, bl + 1, right)
    return

# todo 返回划分值等于区域的 左右边界
def partition(arr, left, right):
    pivot = arr[right]
    sr = left - 1     # todo 没有小于区域的话，会在process的if判断中直接返回。
    bl = right
    i = left
    while(i < bl):
        if arr[i] < pivot:
            sr += 1
            arr[i], arr[sr] = arr[sr], arr[i]
            i += 1
        elif arr[i] == pivot:
            i += 1
        else:
            bl -= 1
            arr[i], arr[bl] = arr[bl], arr[i]
    arr[right], arr[bl] = arr[bl], arr[right]
    return sr + 1, bl

if __name__ == '__main__':
    arr = [4,3,3,4,5,1]
    quickSort(arr)
    print(arr)


# ----------------------------------------------- #
