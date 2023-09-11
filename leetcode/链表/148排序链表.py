# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def sortList(self, head: ListNode) -> ListNode:
        '''
        大函数的定义就是返回排好序的list
        :param head:
        :return:
        '''
        if head is None or head.next is None:
            return head

        # 找中点
        fast, slow = head, head
        while(fast.next is not None and fast.next.next is not None):
            fast = fast.next.next
            slow = slow.next
        head1 = head
        head2 = slow.next
        slow.next = None   # todo 要断开的，很关键！

        head1 = self.sortList(head1)
        head2 = self.sortList(head2)
        cur1 = head1
        cur2 = head2

        dummy_head = ListNode(0)
        cur = dummy_head
        # 合并
        while(cur1 is not None and cur2 is not None):
            if cur1.val < cur2.val:
                cur.next = cur1
                cur1 = cur1.next
            else:
                cur.next = cur2
                cur2 = cur2.next
            cur = cur.next
            cur.next = None
        if cur1 is not None:
            cur.next = cur1
        else:
            cur.next = cur2

        new_head = dummy_head.next
        dummy_head.next = None
        return new_head

    # 归并排序那一套
    def sortList1(self, head: ListNode) -> ListNode:
        # 排序
        def process(head):
            if head is None or head.next is None:
                return head

            # 核型还是找中点
            slow, fast = head, head
            while (fast.next and fast.next.next):
                slow = slow.next
                fast = fast.next.next

            head1, head2 = head, slow.next
            slow.next = None
            head1 = process(head1)
            head2 = process(head2)

            return merge(head1, head2)

        def merge(head1, head2):
            if head1 is None:
                return head2
            if head2 is None:
                return head1

            dummy = ListNode()
            cur = dummy

            while (head1 and head2):
                if head1.val < head2.val:
                    cur.next = head1
                    head1 = head1.next
                else:
                    cur.next = head2
                    head2 = head2.next
                cur = cur.next
                cur.next = None
            cur.next = head1 if head1 else head2

            return dummy.next

        return process(head)


# ok了
# 想清楚归并的过程，数组要help存，list本身结构可以O(1)
# 递归要深刻！
# 断开的细节是list排序的关键！

