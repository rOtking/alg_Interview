# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:

        p1 = l1
        p2 = l2
        add_bit = 0  # 进位

        res = ListNode(100)
        cur = res
        while (p1 != None or p2 != None or add_bit != 0):
            if (p1 != None and p2 != None):
                value = p1.val + p2.val + add_bit
                p1 = p1.next
                p2 = p2.next
            elif p1 != None:
                value = p1.val + add_bit
                p1 = p1.next
            elif p2 != None:
                value = p2.val + add_bit
                p2 = p2.next
            else:
                value = add_bit
            if value >= 10:
                value -= 10
                add_bit = 1
            else:
                add_bit = 0

            node = ListNode(value)
            cur.next = node
            cur = node

        return res.next

# ok
# todo 常规操作，双指针移动即可
