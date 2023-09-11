# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if l1 is None:
            return l2
        if l2 is None:
            return l1

        cur1, cur2 = l1, l2
        dummyNode = ListNode(0)
        cur = dummyNode
        # 出去时至少一个是None
        while (cur1 is not None and cur2 is not None):
            if cur1.val < cur2.val:
                cur.next = cur1
                cur = cur.next
                cur1 = cur1.next
            else:
                cur.next = cur2
                cur = cur.next
                cur2 = cur2.next

        cur.next = cur1 if cur2 is None else cur2

        return dummyNode.next


    # def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
    #     if l1 is None:
    #         return l2
    #     if l2 is None:
    #         return l1
    #
    #     dummy = ListNode(0)
    #     cur = dummy
    #     while (l1 and l2):
    #         if l1.val < l2.val:
    #             cur.next = l1
    #             l1 = l1.next
    #             cur = cur.next
    #         else:
    #             cur.next = l2
    #             l2 = l2.next
    #             cur = cur.next
    #
    #     if l1:
    #         cur.next = l1
    #     if l2:
    #         cur.next = l2
    #
    #     return dummy.next


# ok 简单！
# todo 开头边界要细心，被弱智失误困扰好久！😄
