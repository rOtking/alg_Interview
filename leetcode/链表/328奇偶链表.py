# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def oddEvenList1(self, head: ListNode) -> ListNode:
        if head is None or head.next is None or head.next.next is None:
            return head

        # 至少两个节点
        dummy_odd = ListNode()
        dummy_even = ListNode()

        cur = head

        odd_tail = dummy_odd   # 奇的尾节点记录
        even_tail = dummy_even
        while(cur is not None and cur.next is not None):
            # cur.next不是尾节点
            odd = cur
            even = cur.next
            next_ = cur.next.next

            odd.next = None
            even.next = None
            odd_tail.next = odd
            even_tail.next = even

            cur = next_
            odd_tail = odd_tail.next
            even_tail = even_tail.next

        # curNone或cur是尾还没处理
        if cur is not None:
            odd_tail.next = cur
            odd_tail = odd_tail.next

        head1 = dummy_odd.next
        dummy_odd.next = None
        head2 = dummy_even.next
        dummy_even.next = None
        odd_tail.next = head2

        return head1

    # todo 核心就是cur要断开连接！！！避免循环！
    def oddEvenList(self, head: ListNode) -> ListNode:
        if head is None or head.next is None:
            return head

        dummy1 = ListNode()
        dummy2 = ListNode()
        cur1 = dummy1
        cur2 = dummy2
        # 一边删一边加
        cur = head
        switch = True
        while (cur):
            # todo
            next_ = cur.next
            cur.next = None
            if switch:
                cur1.next = cur
                cur1 = cur1.next
            else:
                cur2.next = cur
                cur2 = cur2.next
            cur = next_
            switch = not switch
        cur1.next = dummy2.next
        dummy2.next = None
        return dummy1.next


# ok
# 奇偶、首尾，4个节点搭配好就行了。
