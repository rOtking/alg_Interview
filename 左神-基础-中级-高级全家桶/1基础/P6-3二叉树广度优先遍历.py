'''
todo 二叉树的广度优先搜索
'''

# ------------------------ 1.层序遍历，即广度优先 ------------------ #
'''
核心：使用队列结构！

queue先放入cur=root
（1）cur pop出来；
（2）处理、收集cur；
（3）如果存在left与right，add进queue；
（4）上述3步为loop，queue不空就继续。

注意：左神是用map与几个变量来记录层级的；而我这里用的labuladong的BFS模版，while中的for正好
    天然的将一层的node都处理了。每个while都处理一层，很精妙！
'''

# ok 的
def levelOrder(root) -> List[List[int]]:
    res = []
    if root is None:
        return res
    cur = root
    queue = [cur]
    while(len(queue) != 0):
        tmp = []
        size = len(queue)   # todo 关键点！！
        for _ in range(size):
            cur = queue.pop(0)
            tmp.append(cur.val)
            if cur.left:
                queue.append(cur.left)
            if cur.right:
                queue.append(cur.right)
        res.append(tmp)
    return res

# --------------------------------------------- #