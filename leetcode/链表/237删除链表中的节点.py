# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        # todo 垃圾玩意儿！

        # 不是删结点，而是删值，垃圾题目
        val = node.next.val
        node.next = node.next.next
        node.val = val

        return