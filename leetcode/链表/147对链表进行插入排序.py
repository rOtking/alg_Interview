# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    # def insertionSortList(self, head: ListNode) -> ListNode:
    #     if head is None or head.next is None:
    #         return head
    #
    #     dummy_node = ListNode(0)
    #     dummy_node.next = head
    #     cur = head.next
    #     # 有序区结尾
    #     sort_tail = head
    #     while(cur):
    #         next_ = cur.next
    #         pre = dummy_node
    #
    #         if sort_tail.val <= cur.val:
    #             sort_tail = sort_tail.next
    #         else:
    #             # 这里就一定会在有序list的中间了
    #             while(pre.next.val <= cur.val):
    #                 pre = pre.next
    #
    #             tmp = pre.next
    #             pre.next = cur
    #             cur.next = tmp
    #         sort_tail.next = next_   # todo 注意 cur永远是有序区的下一个，不是head，肯定会被替换的！
    #         cur = next_
    #
    #     new_head = dummy_node.next
    #     dummy_node.next = None
    #     return new_head

    # todo 重写后更精致！
    def insertionSortList(self, head: ListNode) -> ListNode:
        if head is None or head.next is None:
            return head

        dummy = ListNode(-float('inf'))
        dummy.next = head
        cur = head.next
        head.next = None

        while (cur):
            next_ = cur.next
            cur.next = None
            newStart = dummy

            while (newStart.next and newStart.next.val <= cur.val):
                newStart = newStart.next
            # newStart是尾结点或newStart的下一个点是大于当前的点
            # 逻辑同用
            tmp = newStart.next
            newStart.next = cur
            cur.next = tmp

            cur = next_

        return dummy.next

if __name__ == '__main__':
    a = ListNode(-1)
    b = ListNode(5)
    c = ListNode(3)
    d = ListNode(4)
    e = ListNode(0)

    a.next = b
    b.next = c
    c.next = d
    d.next = e

    s = Solution()
    head = s.insertionSortList(a)



# ok
# todo 还是有技巧的，需要重点看一下！