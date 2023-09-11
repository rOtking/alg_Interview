# ------------------ 堆排序的概念 ------------------ #
'''
todo
核心：堆是个思想，逻辑上是完全二叉树，物理实现上是利用完全二叉树的特性在数组上实现的！
堆排序就是不断的堆化，输出堆顶元素，再堆化，反复重复的过程。这样就得到了有序数组。

heapsize很关键：表示从开始的一段连续元素是完全二叉树，超过realsize的是不考虑的。 其实也就给了我们剩下部分自我操作的空间。

heapInsert：从最后一个位置开始，依次与parent比较，一路向上交换的过程。插入的过程
heapify堆化：从任意一个位置开始，与子节点比较，一路向下交换的过程。pop之后的调整过程
heapify三个参数：arr，index，heapSize
            index是从哪个位置开始向下做heapify；heapSize就是认为是完全二叉树的连续元素个数。
            heapSize来控制子节点越界的问题，维护堆的大小。arr的大小不是真实堆的大小。
            todo 注意while中只判断left即可，就能代表是否有孩子了。有右必有左，有一个孩子就可以比较！


todo 堆中最核心的两个操作：heapInsert与heapify，时间代价都是O(logn)
 如果从堆中修改了一个数，只要heapInsert或heapify走一个（有且只能走一个），就能调整为堆，而不需要知道改成了什么数，大还是小。

todo 性质：
 总元素n个，i位置的左孩子left = 2 * i + 1，right = left + 1 = 2 * i + 2
 i位置的parent = (i - 1) // 2
 判断叶子：i < n // 2 就是非叶子

'''
# ------------------------------------------------ #
# —————————————————— 堆排序的流程 —————————————————— #
'''
如 9,4,3,6,8,1
1.初始arr进行堆化，搞成一个大根堆：
    （1）认为一个个add进来，heapSize开始是1，即0-1上的数字也就是0位置的9是大根堆；
    （2）接着heapSize += 1，即0-2上的0与1位置9，4构成大根堆，也就是进行一次heapInsert操作；
    （3）直到heapSize为6时，所有数构成了一个大根堆，位置0-6上都heapInsert完成了。
    9,4,3,6,8,1 -> 9,6,3,4,8,1 -> 9,8,3,4,6,1   完成大根堆构建
2.依次heapify进行排序：
    （1）最后元素与堆顶交换，即变为1,8,3,4,6, 9，并heapSize -= 1,得到heapSize=5，即位置0-5还是维护，位置6是最大值了，就不动了；
        需要进行位置0的heapify，将0-5位置重新调整为大根堆 1,8,3,4,6,  9 -> 8,1,3,4,6,  9 -> 8,6,3,4,1,  9
    （2）继续与堆顶交换，heapSize -= 1，并进行heapify的调整，直到heapSize = 1
3.用大根堆，结束时的顺序就是从小到大！ 


堆化：
    所有从后向前的heapify整体for的过程，称为堆化。
    叶子结点n/2个，操作1次
    倒数第二层 n/4个，操作2次
    ...
    T(n) = n/2 * 1 + n/4 * 2 + n/8 * 3 + ...
    2T(n) = n + n/2 * 2 + n/4 * 3 + ...
    2T(n) - T(n) = T(n) = n + n/2 + n/4 + n/8 + ...
    等比数列求和，收敛于O(n) 
'''

# ------------------------------------------------ #

def heapSort(arr):

    # todo 这种做法是认为一个个加入的做法，总时间O(nlogn)，目的是展示heapInsert的操作
    # for i in range(1, len(arr)):    # O(n)
    #     heapInsert(arr, i)          # O(logn)
    # todo 可替换为从后向前的heapify操作，保证每个结点都是大根堆，最终时间收敛于O(n)，更快一些；
    #  这种是认为一下把所有元素都给了，不是一个个来的。与leecode的讲解就一致了
    for i in range(len(arr) - 1, -1, -1):
        heapify(arr, i, len(arr))      # todo 这里的heapSize = len(arr)很关键!!


    heapSize = len(arr) - 1
    while(heapSize > 0):            # O(n)
        arr[0], arr[heapSize] = arr[heapSize], arr[0]
        heapify(arr, 0, heapSize)   # O(logn)
        heapSize -= 1

    print(arr)
    return arr


# todo index表示当前要加入的数在index位置上
# 回顾list可变类型的函数传递问题，python赋值操作全是引用传递。函数里改了，外面也能访问的到！
def heapInsert(arr, index):
    parent = int((index - 1) / 2)
    # todo 细节：应该还有一个条件就是index为0是就停止，因为没有parent
    #  这里0的parent还是0，arr[0] > arr[0]是不会成立的，所以会退出！一举两得！
    while(arr[index] > arr[parent]):
        arr[index], arr[parent] = arr[parent], arr[index]
        index = parent
        parent = int((index - 1) / 2)

# todo 从index开始，能不能向下与子节点交换下去
def heapify(arr, index, heapSize):
    left = 2 * index + 1

    # 有子节点且不越界
    while(left < heapSize):
        # 先找孩子中最大值
        right = left + 1
        largest = right if right < heapSize and arr[left] < arr[right] else left

        # 再与当前index比较
        largest = index if arr[index] > arr[largest] else largest

        # 比孩子大
        if index == largest:
            break

        arr[index], arr[largest] = arr[largest], arr[index]
        index = largest
        left = index * 2 +1





if __name__ == '__main__':
    heapSort([9,8,7,6,5,4,3])