# ------------------- 0.堆的判断 ------------------- #
'''
以小顶堆为例
1.是完全二叉树：从上到下，从左到右依次填满
2.所有父节点大于子结点
'''
# ------------------------------------------------- #

# ---------------- 1.堆的元素插入过程 ---------------- #
'''
1.放到堆的最后
2.依次与父节点比较，如果小于父节点则与父节点交换
3.重复步骤2，直到大于等于父结点，停止。
'''
# ------------------------------------------------- #

# -------------- 2.堆的堆顶元素删除过程 --------------- #
'''
1.删除堆顶，把最后一个元素放到堆顶并删除最后一个元素
（上述满足了完全二叉树，下面完成第二个条件即可）
2.从堆顶开始，依次与两个子节点比较，与比自己小的且是最小的那个子节点进行交换
3.重复步骤2，直到它小于两个子节点，停止。
'''
# ------------------------------------------------- #

# ------------- 3.完全二叉树转数组及特性 -------------- #
'''
1.二叉树从上到下，从左到右的index编号是从1，2，3...开始
  还将他们依次放到数组arr的对应index位置即可。0位置放上元素个数
  
2.位置n的父节点：n/2；左孩子：2n；右孩子：2n + 1。
3.位置n是不是叶子节点：如果n > 元素个数 /2，就是叶子节点
'''

# ------------------------------------------------- #

# ----------------- 4.最小堆的实现 ------------------ #
class MinHeap:
    # 初始化：固定容量的堆
    def __init__(self, heapSize):
        self.heapSize = heapSize
        self.minHeap = [0] * (heapSize + 1)
        self.realSize = 0    # 真实的元素个数

    # 最简单的，到达容量就不能再加入了！
    def add(self, ele):
        self.realSize += 1
        if self.realSize > self.heapSize:
            print('超容量了，先pop，再add。')
            self.realSize -= 1
            return
        self.minHeap[self.realSize] = ele
        # 索引位置
        index = self.realSize

        # todo 细节 a // b 与 int(a / b)一样的
        parent = index // 2
        while(self.minHeap[index] < self.minHeap[parent]):
            self.minHeap[index], self.minHeap[parent] = self.minHeap[parent], self.minHeap[index]
            index = parent
            parent = index // 2

        return

    def peek(self):
        return self.minHeap[1]

    def pop(self):
        if self.realSize < 1:
            print('当前没有元素！')

        # 下面就是有元素的情况了
        removeEle = self.minHeap[1]
        self.minHeap[1] = self.minHeap[self.realSize]
        self.realSize -= 1

        index = 1

        # 子节点就停了，注意范围啊！
        while(index < self.realSize and index <= self.realSize // 2):
            left = 2 * index
            right = 2 * index + 1
            if self.minHeap[index] < self.minHeap[left] or self.minHeap[index] < self.minHeap[right]:
                if self.minHeap[left] < self.minHeap[right]:
                    self.minHeap[index], self.minHeap[left] = self.minHeap[left], self.minHeap[index]
                    index = left
                else:
                    self.minHeap[index], self.minHeap[right] = self.minHeap[right], self.minHeap[index]
                    index = right

        return removeEle

    # 返回堆的真实元素个数
    def size(self):
        return self.realSize

    def toString(self):
        print(self.minHeap[1: self.realSize + 1])
# ------------------------------------------------- #

if __name__ == "__main__":
        # 测试用例
        minHeap = MinHeap(5)
        minHeap.add(3)
        minHeap.add(1)
        minHeap.add(2)
        # [1,3,2]
        minHeap.toString()
        # 1
        print(minHeap.peek())
        # 1
        print(minHeap.pop())
        # 2
        print(minHeap.pop())
        # 3
        print(minHeap.pop())
        minHeap.add(4)
        minHeap.add(5)
        # [4,5]
        minHeap.toString()

# todo ok没问题！



