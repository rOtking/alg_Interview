# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    # 递归法
    def reverseN(self, head, n: int):
        '''
        反转前n的节点
        :param n:
        :return:
        '''
        if n == 1:
            return head

        new_head = self.reverseN(head.next, n - 1)
        new_last = head.next    # 反转部分的尾
        next = new_last.next  # 记录一下不反转的部分
        head.next = next
        new_last.next = head

        return new_head

    def reverseBetween(self, head: ListNode, left: int, right: int) -> ListNode:

        if left == 1:
            head = self.reverseN(head, right)
            return head

        head.next = self.reverseBetween(head.next, left - 1, right - 1)
        return head

    # 迭代法
    def reverseList(self, head):
        if head is None or head is None:
            return head

        pre = None
        cur = head
        while (cur):
            next_ = cur.next
            cur.next = pre
            pre = cur
            cur = next_
        return

    def reverseBetween1(self, head: ListNode, left: int, right: int) -> ListNode:
        if left == right:
            return head

        # 找left - 1与right+1的位置
        dummy = ListNode()
        dummy.next = head
        # 左半的尾，右半的头
        leftTail, rightHead = dummy, None
        # 需要翻转的头尾
        reverseHead, reverseTail = head, None
        i = 0
        cur = dummy
        while (cur):
            if i == left - 1:
                leftTail = cur
                reverseHead = cur.next
            if i == right:
                reverseTail = cur
                rightHead = cur.next
            cur = cur.next
            i += 1
        reverseTail.next = None
        self.reverseList(reverseHead)
        leftTail.next = reverseTail
        reverseHead.next = rightHead

        return dummy.next


# ok
# 转为反转前n个数，作为终止条件
# 注意范围是一直变化的！






