# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    # 迭代版本
    def reverseList1(self, head: ListNode) -> ListNode:

        # 找个变量把断开的地方存起来就行了，防止找不到
        if head is None or head.next is None:
            return head

        pre = None
        cur = head
        next = cur.next

        while(next is not None):
            cur.next = pre

            pre = cur
            cur = next
            next = next.next

        # 此时1-2-3-4 5，pre是4，cur是5，next是None
        # 还要在连接一次
        cur.next = pre

        return cur

    # 递归版本
    def reverseList(self, head: ListNode) -> ListNode:
        '''
        核心是搞清函数的定义：接收头节点，返回反转后的头节点
        :param head:
        :return:
        '''
        if head is None or head.next is None:
            return head

        # 大循环
        new_head = self.reverseList(head.next)
        # 新list的尾节点
        new_last = head.next

        head.next = None
        new_last.next = head

        return new_head




# ok
# 注意None的特殊情况就没什么别的问题了！









