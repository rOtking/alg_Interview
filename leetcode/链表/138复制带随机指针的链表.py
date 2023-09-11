
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random


class Solution:
    def __init__(self):
        self.tables = {}

    def copyRandomList(self, head: 'Node') -> 'Node':
        new_head = None
        if head is None:
            return new_head

        if head in self.tables:
            return self.tables[head]



        new_head = Node(head.val)

        self.tables[head] = new_head
        new_head.next = self.copyRandomList(head.next)
        new_head.random = self.copyRandomList(head.random)

        return new_head

    # 迭代法
    # todo 关键点：map1存原来-新，map2寸新-原来   空间O(n)
    def copyRandomList1(self, head: 'Node') -> 'Node':
        if head is None:
            return head
        newHead = ListNode(head.val)
        # map1存原来-新，map2寸新-原来
        map1, map2 = {}, {}
        map1[head] = newHead
        map2[newHead] = head
        cur1 = head
        cur2 = newHead
        while (cur1.next):
            node = Node(cur1.next.val)
            cur2.next = node
            cur1 = cur1.next
            cur2 = cur2.next
            map1[cur1] = cur2
            map2[cur2] = cur1

        cur1 = head
        cur2 = newHead
        while (cur2):
            cur2.random = map1[map2[cur2].random] if map2[cur2].random else None
            cur1 = cur1.next
            cur2 = cur2.next

        return newHead
# todo 法三是   node1-copy1-node2-copy2的方法
#  copy1的randoms是node1的random的next！

# 明确函数定义，复制以head为头的list，包括next与random，random会断但是next能保证不丢失
#核心是dict来存建立过的node！！！todo 继续理解！