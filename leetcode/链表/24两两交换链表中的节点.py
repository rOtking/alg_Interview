# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    # def swapPairs(self, head: ListNode) -> ListNode:
    # if head is None or head.next is None:
    #     return head

    # pre = None
    # cur = head
    # head = cur.next

    # while(cur is not None):

    #     # 两个数字，要反转
    #     if cur.next is not None:
    #         cur2 = cur.next
    #         suf = cur.next.next  # 可能是None   todo 注意处理！
    #         # 反转cur与cur2
    #         cur2.next = cur
    #         cur.next = None
    #     # 一个数字 不反转
    #     else:
    #         cur2 = cur
    #         suf = None
    #     # 与前面相连
    #     if pre is not None:
    #         pre.next = cur2

    #     # update
    #     pre = cur
    #     cur = suf

    # return head


    # todo 改进版  ok的！哈哈

    def swapPairs(self, head: ListNode) -> ListNode:
        # 反转两个node
        def reverseTwoNode(head):
            if head is None or head.next is None:
                return head
            next_ = head.next
            next_.next = head
            head.next = None
            # 返回头尾
            return next_, head

        dummy = ListNode()
        dummy.next = head
        pre = dummy
        cur = head

        while (cur and cur.next):
            next_ = cur.next.next  # 可能是None
            h, t = reverseTwoNode(cur)
            pre.next = h
            pre = t
            cur = next_

        pre.next = cur

        return dummy.next

'''
迭代也简单。

这里是定义一个两node交换的函数。遍历list依次跳两个即可。注意边界条件

'''












