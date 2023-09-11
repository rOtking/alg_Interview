# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    # def hasCycle(self, head: ListNode) -> bool:
    # if head is None or head.next is None:
    #     return False

    # index1 = head
    # index2 = head

    # while(index2 is not None):
    #     index1 = index1.next

    #     if index2.next is None:
    #         return False
    #     else:
    #         index2 = index2.next.next

    #     if index1 == index2:
    #         return True

    # return False

    # todo 新写的精致多了！
    def hasCycle(self, head: ListNode) -> bool:
        if head is None or head.next is None:
            return False

        slow, fast = head, head

        while (fast and fast.next):
            slow = slow.next
            fast = fast.next.next
            if slow is fast:
                return True

        return False