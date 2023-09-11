# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    # ok
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        if headA is None:
            return headB
        if headB is None:
            return headA

        cur = headA
        lengthA = 1
        while (cur.next):
            lengthA += 1
            cur = cur.next
        cur = headB
        lengthB = 1
        while (cur.next):
            lengthB += 1
            cur = cur.next

        longNode = headA if lengthA > lengthB else headB
        shortNode = headB if longNode is headA else headA

        for _ in range(abs(lengthA - lengthB)):
            longNode = longNode.next

        while (longNode):
            if longNode is shortNode:
                return longNode
            longNode = longNode.next
            shortNode = shortNode.next
        return None










