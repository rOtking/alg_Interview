# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    # def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
    # if head is None or n < 1:
    #     return head

    # left = head
    # right = head

    # for _ in range(n -1):
    #     if right.next is None:
    #         return head
    #     right = right.next

    # # 走到待删除前一个节点
    # if right.next is not None:
    #     right = right.next
    # else:
    #     return head.next

    # while(right.next is not None):
    #     left = left.next
    #     right = right.next

    # node1 = left.next
    # node2 = node1.next

    # left.next = node2
    # node1.next = None

    # return head


    # 已知至少一个结点
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        slow, fast = head, head
        # 删除倒数n，需要找倒数n+1
        for _ in range(n):
            fast = fast.next
        # 删除的是head
        if fast is None:
            newHead = head.next
            head.next = None
            return newHead

        while (fast.next):
            slow = slow.next
            fast = fast.next

        # slow是倒数n+1
        removeNode = slow.next
        next_ = removeNode.next
        removeNode.next = None
        slow.next = next_

        return head



# ok
'''
找倒数n+1的结点方便删除

（1）倒数n是head
（2）不是head

'''






