# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    # 逆序 边界太多了！
    def plusOne1(self, head: ListNode) -> ListNode:
        if head.next is None:
            val = head.val
            val += 1
            if val > 9:
                newHead = ListNode(1)
                head.val = val - 10
                newHead.next = head
                return newHead
            else:
                head.val = val
                return head

        newHead = self.reverseList(head)
        addBit = 1
        cur = newHead
        while (cur.next):
            next_ = cur.next
            val = cur.val + addBit
            if val > 9:
                val = val - 10
                addBit = 1
            else:
                addBit = 0
            cur.val = val
            cur = cur.next
        if addBit == 1:
            val = cur.val + 1
            if val <= 9:
                cur.val = val
            else:
                cur.val = val - 10
                cur.next = ListNode(1)
        return self.reverseList(newHead)

    def reverseList(self, head):
        if head is None or head.next is None:
            return head
        pre = None
        cur = head
        while (cur):
            next_ = cur.next
            cur.next = pre
            pre = cur
            cur = next_
        return pre

    # 借助stack 辅助数组 更加直观！
    def plusOne(self, head: ListNode) -> ListNode:
        stack = []
        cur = head
        while (cur):
            stack.append(cur)
            cur = cur.next
        addBit = 1
        for i in range(len(stack) - 1, -1, -1):
            val = stack[i].val + addBit
            if val > 9:
                val -= 10
                addBit = 1
            else:
                addBit = 0
            stack[i].val = val

        if addBit == 1:
            newHead = ListNode(addBit)
            newHead.next = head
            return newHead
        return head

# ok