# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    # def removeElements(self, head: ListNode, val: int) -> ListNode:
    #     if head is None:
    #         return head
    #     dummy_head = ListNode(0)
    #     dummy_head.next = head
    #
    #     pre = dummy_head
    #     cur = pre.next
    #     next_ = cur.next
    #
    #     while(next_ is not None):
    #         if cur.val == val:
    #             pre.next = next_
    #             cur.next = None
    #
    #             cur = next_
    #             next_ = next_.next
    #         else:
    #             pre = cur
    #             cur = next_
    #             next_ = next_.next
    #
    #     # 此时next_ 为None，也就是cur是尾节点没处理
    #     if cur.val == val:
    #         pre.next = None
    #
    #     new_head = dummy_head.next
    #     dummy_head.next = None
    #     return new_head

    # todo 精简了好多，上面是什么垃圾啊！！！
    def removeElements(self, head: ListNode, val: int) -> ListNode:
        dummy = ListNode()
        dummy.next = head

        pre = dummy
        cur = head
        while(cur):
            next_ = cur.next
            if cur.val == val:
                pre.next = next_
            else:
                pre = cur
            cur = next_
        return dummy.next

# ok了！
# 哑节点是真好用的啊！