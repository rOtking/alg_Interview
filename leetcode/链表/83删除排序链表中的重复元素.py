# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:

    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if head is None or head.next is None:
            return head

        pre = head
        cur = head.next
        while (cur):
            if pre.val == cur.val:
                cur = cur.next
            else:
                pre.next = cur
                pre = cur
                cur = cur.next
        # pre就是最后一个重复（或不重复）的首个数字
        pre.next = None

        return head




# ok
