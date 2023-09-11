'''

'''
'''
BFS与DFS是一种搜索遍历的思想，用于图上，可解决有环的问题！二叉树当然也能用啦，BT还不存在有环的重复问题呢！

BFS：
（1）用queue实现；
（2）从源结点开始进队列，然后pop；
（3）pop一个，处理一个，并把与之相连的没进过队列的邻接点放入队列；
（4）直到queue空。

有没有进过队列用set来维护，因为有环，所以进过的就不再进了，不然会没完了！

实质：就是每次都尝试所有可能，这样就能找到最短路径了！


            A
          / | \ 
         B  C--D
            |  /
             E
BFS：

(1)queue:[A],set:{A}
(2)queue pop A,queue:[],set:{A}
(3)处理A,A的nexts:BCD不在set,入queue，在set注册    queue:[B,C,D]   set:{A,B,C,D}
(4)queue pop B,queue:[C,D]
(5)处理B,B没有nexts
(6)queue pop C,queue:[D]
(7)处理C,C的nexts是ADE，AD在set，不重复加入；E不在set，   queue[D,E]   set:{A,B,C,D,E}
(8)queue pop D,queue:[E]
(9)处理D,D的nexts为ACE，都注册过
(10)queue pop E,queue:[]
(11)处理E，E的nexts为CD，注册过。
(12)遍历结束

顺序为   ABCDE

模版
'''

def bfs(node):

    queue = [node]      # 处理列表
    visited = set(node)  # 注册表
    while(len(queue) != 0):
        cur = queue.pop(0)
        # todo 对cur的相关操作

        # todo 遍历所有邻接点
        for nextNode in cur.nexts:
            if nextNode not in visited:
                queue.append(nextNode)
                visited.add(nextNode)
'''
上面这种方式，某一时刻queue中的元素并不是同一个step的。因为有新的子节点加入queue；

下面这种加一层循环限制，能保证每次的queue都是同一个step上的全量！
注意提前存下size，不能用len(queue)，因为子过程是在加入queue的。

'''
def bfs1(node):
    queue = [node]
    visited = set(node)
    while(len(queue) != 0):
        size = len(queue)
        for _ in range(size):
            cur = queue.pop(0)
            for nextNode in cur.nexts:
                if nextNode not in visited:
                    queue.append(nextNode)
                    visited.add(nextNode)






'''
DFS:
（1）用stack实现
（2）源结点放入stack,处理
（3）stack pop结点，如果有没进过stack的邻接点，stack 先push当前node保存现场，再（只）push其中一个没进过的邻接点
（4）直到stack空



实质：stack中存的就是dfs的一条路径；

那当前结点重复压入就是为了保存之前的现场，以便回溯后不遗漏的继续新的dfs；

break的作用就是保证一条路走到底，先不走同级别的其他点，因为cur也在stack中，同级别的其他点会在后序访问到。

            A
          / | \ 
         B  C--D
            | /
            E
             
DFS：
(1)stack push stack:[A] ,处理A, set:{A}
(2)stack pop A, A的nexts:BCD, 依次选没注册过的 先B
(3)B不在set,stack push A push B  ,处理B, stack:[A,B],注册set:{A,B}
(4)stack pop B, B的nexts:A,已注册
(5)stack pop A, A的nexts:BCD，CD没注册，选C
(6)stack push A, push C, 处理C, stack:[A,C], set:{A,B,C}
(7)stack pop C, C的nexts:ADE, DE没注册，选D
(8)stack push C, push D, 处理D, stack:[A, C, D], set:{A,B,C,D}
(9)stack pop D, D的nexts:ACE,  E
(10)stack push D, push E, 处理E, stack:[A,C,D,E], set:{A,B,C,D,E}
(11)stack pop E, E的nexts:CD 已注册
(12)stack pop D, D的nexts:ACE 已注册
(13)stack pop C, C的nexts:ADE 已注册
(14)stack pop A, A的nexts:BCD 已注册
(15)stack空 结束


处理顺序为   ABCDE  当然可以是其他的，取决于选取同级别nexts时的顺序。

'''

def dfs(node):
    stack = [node]
    visited = set(node)
    # todo 处理node
    while(len(stack) != 0):
        cur = stack.pop()
        for nextNode in cur.nexts:
            if nextNode not in visited:
                stack.append(cur)
                stack.append(nextNode)
                visited.add(nextNode)
                # todo 处理node
                break
'''
当然dfs可以递归实现，stack就是在模拟递归保存现场的实现。

模版如下：

'''
def backtrack(track, candidates):
    # 1.终止条件
    if False:
        return

    # 2.一顿操作，确定当前的可选择列表
    # track是做过的选择，candidates是当前可做的选择

    # 3.做选择
    for candidate in candidates:
        # 更新路径
        track.append(candidate)

        newCandidates = func(candidates)   # todo 对应的新候选列表
        # 继续走到底
        backtrack(track, newCandidates)

        # 恢复现场
        track.pop()


'''
二叉树的DFS
                1
              /   \ 
             2     3
            / \   /  
           4   5 6
          /
         7
    1 2 4 7.... 1 2 5... 1 3 6
    
    尴尬 BT的DFS就是前序遍历。
'''
def btDFS(root):
    stack = [root]
    visited = set()
    visited.add(root)
    print(root.val)

    while(len(stack) != 0):
        cur = stack.pop()
        if cur.left and cur.left not in visited:
            stack.append(cur)
            stack.append(cur.left)
            visited.add(cur.left)
            print(cur.left.val)
            continue
        if cur.right and cur.right not in visited:
            stack.append(cur)
            stack.append(cur.right)
            visited.add(cur.right)
            print(cur.right.val)
            continue

# todo  ok了，理解深了一层！
def btDFS1(root):
    # todo 尝试递归版本的二叉树DFS
    # 终止条件
    if root is None:
        return
    print(root.val)
    # 构建新candidates：就是lr,直接做选择
    if root.left:
        btDFS1(root.left)
    # 相当于pop后再选择，bt的特性使它不用搞一个track进行push与pop
    if root.right:
        btDFS1(root.right)




# todo 左神的版本适合BT的DFS，我的貌似就是适合非BT的回溯。


class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

node1 = TreeNode(1)
node2 = TreeNode(2)
node3 = TreeNode(3)
node4 = TreeNode(4)
node5 = TreeNode(5)
node6 = TreeNode(6)
node7 = TreeNode(7)

node1.left = node2
node1.right = node3
node2.left = node4
node2.right = node5
node3.left = node6
node4.left = node7

btDFS1(node1)


