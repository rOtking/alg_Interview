'''
归并就是递归过程！

'''

# ------------------- 1.归并的过程 ------------------ #
'''
[左侧, 右侧]

最外层过程 + 递归过程 + merge过程    3部分
大递归
1.默认左侧排好序，右侧排好序。（子过程的排序是递归实现的）
2.两部分外排，merge起来。
3.处理好终止条件即可。

依然是 二叉树 后序遍历结构：左孩子完成排序，右孩子完成排序，最终访问父节点的位置实现merge。则这棵子树就完成了归并排序。

'''
# ------------------------------------------------- #


# ------------------- 2.代码实现 ------------------- #
def mergeSort(arr):
    process(arr, 0, len(arr) - 1)
    print(arr)

# 递归过程 ：完成left到right上的排序
def process(arr, left, right):
    if left == right:
        return
    mid = left + ((right - left) >> 1)
    process(arr, left, mid)
    process(arr, mid + 1, right)
    merge(arr, left, mid, right)
    return

# 合并过程
# todo 都是list函数参数引用传递的应用，list在函数内变化，外面有感知！
def merge(arr, left, mid, right):        # O(n)
    help_arr = []
    p1 = left
    p2 = mid + 1
    while(p1 <= mid and p2 <= right):
        if arr[p1] <= arr[p2]:
            help_arr.append(arr[p1])
            p1 += 1
        else:
            help_arr.append(arr[p2])
            p2 += 1

    while(p1 <= mid):
        help_arr.append(arr[p1])
        p1 += 1

    while(p2 <= right):
        help_arr.append(arr[p2])
        p2 += 1
    # 赋值
    for i in range(left, right + 1):
        arr[i] = help_arr[i - left]

# ok
if __name__ == '__main__':
    arr = [5,4,3,6,2,1]
    mergeSort(arr)

# ------------------------------------------------- #

# ---------------------- 3.复杂度 -------------------- #
'''
T(N) = 2 * T(N / 2) + O(N)
a = 2, b = 2, d = 1

log(b, a) = 1 = d

T(N) = O(N ^ d * logN) = O(NlogN)


空间上因为有help数组，O(N)：最多准备N的空间就够了，可以释放了循环利用的！
'''
# ------------------------------------------------- #

# ---------------------- 4.分析比O(N^2)好在哪里 -------------------- #
'''
O(N^2)浪费了大量的比较行为：0 .... N-1，比较N-1次才确定了0位置，再比较N-2次才确定1位置
0 ....... N-1
  1 ..... N-1
    2.... N-1
      ...
      
除了每轮确定的一个位置，其他位置的比较信息都丢弃了！没有利用起来！

归并：左部分，右部分      每次比较合并成一个新的有序部分，所有比较的信息是保留下来并传递下去的！
    merge后的部分作为一个有序部分了，在后续过程中他们的相对位置就不会改变了！
    所以时间自然会节省一些！


'''

