'''
后继结点：中序遍历中的下一个node就是后继。

普通遍历找后继是O(n)所有结点的，如果有parent指针，假设node2是node1的后继，距离是k

那时间可以做到O(k)!
'''

'''
node的后继
方法还是分情况讨论后继：
（1）node有right，那就是right子树上的最左；
（2）node无right，node的parent链上，node在某个parent的左，那就是这个parent。
           a 
          / \ 
         o   x
          \  ...
           o 
            \ 
             node

node一直找到是a的左孩子，那a就是后继。

细节边界：BT的最后一个结点的后继是None！
'''
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None

def getSuccessorNode(node):

    if node.right:
        # 找最左 cur非空
        cur = node.right
        while(cur.left):
            cur = cur.left
        return cur
    cur = node.parent
    while(cur is not None and node is cur.right):
        node = cur    # todo 注意 node也要更新，不然不能确定父子关系！
        cur = cur.parent
    return cur

