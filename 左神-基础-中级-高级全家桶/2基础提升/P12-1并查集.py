''''''
# --------------------- 1。岛问题 --------------------#
'''
0 1 0 0 1 0
1 1 1 0 1 0
0 1 1 0 0 0
0 0 0 1 0 0
每个位置上下左右的1连起来形成岛，斜线不算，求岛数量？

就是递归的感染过程infect，递归的检查每个位置的上下左右，是就改成2。类似BFS。

'''
def islandNum(map):
    # 包含ij位置的岛
    def infect(i, j, map):
        if i < 0 or i >= len(map) - 1 or j <0 or j>= len(map[0]) - 1 or map[i][j] != 0:
            return

        # todo 保证不重复
        map[i][j] = 2

        # 递归
        infect(i - 1, j, map)
        infect(i + 1, j, map)
        infect(i, j - 1, map)
        infect(i, j + 1, map)

        return res

    res = 0
    for i in range(len(map)):
        for j in range(len(map[0])):
            # todo 就一个，感染过就不会算了
            res += 1
            infect(i ,j, map)

    return res


# todo ok
# res = islandNum([
#     [0, 1, 0, 0, 1, 0],
#     [1, 1, 1, 0, 1, 0],
#     [0, 1, 1, 0, 0, 0],
#     [0, 0, 0, 1, 0, 0]
# ])
#
# print(res)

# --------------------------------------------- #
# --------------------- 2.并查集 --------------------#
'''
如果map很大呢？怎么并行计算？-------分成小区域并行，然后合并起来。

并查集是支持快速集合合并的结构。两个操作  isSameSet与union

假设：
1.集合用list表示，那合并操作O(1)，但是查找的是O(n)；
2.用map表示，查O(1)，但是合并的要一个个合并。


并查集做法：
初始几个ele，各自形成一个集合。生成类似有向图，指向自己。
      _      _      _      _      _
     | v    | v    | v    | v    | v
     |_a    |_b    |_c    |_d    |_e


（1）isSameSet：一直找到自己的代表元素，代表一样就在一个集合；如a的代表就是a，b的代表就是b，不在一个集合；
（2）union：先isSameSet，不是一个集合后，数量少的集合的代表挂到多的去，如ab合并，b挂a；

      _        _      _      _
     | v      | v    | v    | v
     |_a      |_c    |_d    |_e
       ^
       |
       b

（3）isSameSet(a,b)：代表都是a，在；
（4）union(b,e):则e挂在b的代表a下；注意不是b下，目的是让集合尽量扁平！
    union(c,d)

      _        _      
     | v      | v      
     |_a      |_c      
       ^        ^
     / |        |
    e  b        d

（5）union(b,d)：c挂a下
      _              
     | v            
     |_a            
       ^      
     / | \       
    e  b  c
          ^
          |
          d 
          
          
查找的优化！
         o 
         a
        / \ 
       b   ...
      / \ 
     x   ...
    / \ 
   y  ..
  / \ 
...  ...

isSameSet(y,?)：y在找代表的时候经历y-x-b-a  ，把这条链的node都变成扁平的。也就是x的父直接a，y也是
                即   x-a，y-a
                这样查的时候不是更快了么？！！哈哈
    ____            
   |    \ o 
   |   -- a
   |  |  / \ 
   |  | b   ...
   |  |  \ 
   |  x   ...
   |   \ 
   y  ..
  / \                 
                
即     a是中心点的放射状。    
            o 
            a - ... 
          / | \ 
         x  b  y
         |  |  |
        ..  .. ..
        
实质：这个结构的缺点就是链可能太长，所以每次查时发现链很长，就把它与代表直连，O(1)的查找。   

结论：UFS的findHead平均复杂度O(1)，所以isSameSet与union平均就是O(1)              
'''
# 用户的数据
class Node:
    def __init__(self, value):
        self.value = value
        # todo 等

# todo UFS要求一定要初始化！元素不会增减
# UFS 并查集结构
class UnionFindSet:
    # 接受一个结点的列表
    def __init__(self, nodeList):
        self.fatherMap = {}    # 查询代表结点的映射    node-父亲
        self.sizeMap = {}     # 代表元素所在的集合有几个点    只有代表元素有记录

        # s=初始化
        for node in nodeList:
            self.fatherMap[node] = node
            self.sizeMap[node] = 1

    def findHead(self, element):
        # 把路径存下来，便于扁平化处理
        stack = []

        # 自身与父一样，就是代表结点
        while(element is not self.fatherMap[element]):
            stack.append(element)
            element = self.fatherMap[element]

        # 此时ele就是代表   stack是除了代表之外的所有点  先扁平化
        while(len(stack) != 0):
            self.fatherMap[stack.pop()] = element

        return element

    def isSameSet(self, a, b):
        if a in self.fatherMap and b in self.fatherMap:
            return self.findHead(a) is self.findHead(b)
        # 初始注册过的元素才去查，否则直接False
        return False
    def union(self, a, b):
        # 注册过才合并
        if a in self.fatherMap and b in self.fatherMap:
            aH = self.findHead(a)
            bH = self.findHead(b)

            if aH is not bH:
                # 判断谁长 谁挂谁
                big = aH if self.sizeMap[aH] >= self.sizeMap[bH] else bH
                small = bH if aH is big else aH

                self.fatherMap[small] = big
                self.sizeMap[aH] = self.sizeMap[aH] + self.sizeMap[bH]
                self.sizeMap.pop(bH)



# --------------------------------------------- #
# --------------------- 3.岛问题加速 --------------------#
'''
一般顶级公司考，就是讲思路。如两个cpu怎么加速？
     :
1 1 1: 1 1
1 0 0: 0 1
1 0 1: 1 1
1 0 0: 0 1
1 1 1: 1 1
     :
本来是一个岛，如果分开求，左边2，右边1。计算时需要记录下边界位置属于哪个岛，再检查边界去掉相连的重复岛。但是hash没有UFS快！

将边界的所属区分
     :
1 1 a: c 1
1 0 0: 0 1
1 0 b: c 1
1 0 0: 0 1
1 1 a: c 1
     :
初始res=3，检查边界：
（1）ac不是一个集合,union(a,c),res-=1;
（2）bc不是一个集合,union(b,c),res-=1;
（3）ac是一个集合

最终res=1，就是一个岛。


其实就是mapReduce思想！

把大数据的问题先分为一个个map，多cpu并行处理；完事结合的过程reduce，UFS是高级的reduce方式！
UFS能快速确定是不是一个集合，快速合并！
'''


# --------------------------------------------- #
