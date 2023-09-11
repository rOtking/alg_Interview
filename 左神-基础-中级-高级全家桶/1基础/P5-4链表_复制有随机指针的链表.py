'''
rand的复制
'''
'''
1.空间O(n)是哈希表  <原node, 新node>即可
注意额外空间是不考虑输入输出的，注意，还包括输出！！！！

2.空间 O(1)
            None
            ^
            |
node1----->node2------>node3------>None
  |         ^            ^ |
  |_________|____________| |
            |______________|
  
有个rand指针

技巧：node后连个node'来copy原始行为

                         None        None
                         ^           ^
                         |           |
node1----->node1'----->node2------>node2'------>node3------>node3'------>None
  |          |           ^           ^           ^ |         ^ |
  |__________|___________|___________|___________| |         | |
             |           |___________|_____________|         | |
             |_______________________|_______________________| |
                                     |_________________________|
核心：node1'->rand是node1.rand.next
然后将copy与原来的分离即可。

（1）按next在copyNode连进去
（2）复制random
（3）分离
'''
"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""

def copyRandomList(head):
    if head is None:
        return None
    # 第一次遍历，串copy的next
    cur = head
    while (cur is not None):
        next_ = cur.next
        cur.next = Node(cur.val)
        cur.next.next = next_
        cur = next_
    # 第二次遍历，连random
    cur = head
    while (cur is not None and cur.next is not None):
        copyNode = cur.next
        # todo 注意为None的时候单独处理，不然报错
        copyNode.random = cur.random.next if cur.random else None
        cur = cur.next.next

    # 分离
    head1, cur1 = head, head
    head2, cur2 = head.next, head.next
    # 保证不是尾结点
    # todo 细节应该能优化，再说吧
    while (cur1 is not None and cur1.next.next is not None):
        # 有1必有2，但是next可能是None
        cur2 = cur1.next
        next_1 = cur2.next
        next_2 = next_1.next

        cur1.next = next_1
        cur2.next = next_2

        cur1 = next_1
        cur2 = next_2
    # 出来时cur1.next.next是None，也就是cur1与cur2是结尾了
    cur1.next = None
    cur2.next = None
    return head2




