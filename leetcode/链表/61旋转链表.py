# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def rotateRight(self, head: ListNode, k: int) -> ListNode:
        if head is None or head.next is None:
            return head
        # 计算长度
        cur = head
        length = 0   # 至少1
        while(cur.next is not None):
            length += 1
            cur = cur.next
        length += 1
        # 此时cur就是尾节点,它的next是None
        # 连成循环
        cur.next = head

        k = k % length
        # 此时length > k
        num = length - k    # 数值上，head向后num个，就是新的head位置
        end = head
        for _ in range(num - 1):
            end = end.next
        start = end.next
        end.next = None
        return start

    # 就是找倒数第k+1个node，与倒数k之后的断开重连到head上
    # k可能很大
    def rotateRight1(self, head: ListNode, k: int) -> ListNode:
        if head is None or head.next is None:
            return head

        cur = head
        i = 0
        while (cur):
            cur = cur.next
            i += 1
        real = k % i
        if real == 0:
            return head
        # 找倒数real+1的位置
        slow, fast = head, head
        for _ in range(real):
            fast = fast.next
        while (fast.next):
            slow = slow.next
            fast = fast.next

        newHead = slow.next
        fast.next = head
        slow.next = None
        return newHead


'''
上面连为循环做，下面找到倒数k+1的node
'''
# ok
# 简单！
# 去掉不必要的整倍循环，搞清计算head节点的数值关系就行了！



