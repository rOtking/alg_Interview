# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    # todo 递归大法好！对递归开始有点感觉了！哈哈 加油
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if head is None or head.next is None:
            return head

        # 寻找第一个不同的节点
        key = head.val
        cur = head
        while(cur.next is not None and cur.next.val == key):
            cur = cur.next

        # 判断是否重复
        if head is cur:
            # 不重复
            head.next = self.deleteDuplicates(cur.next)
            return head
        else:
            return self.deleteDuplicates(cur.next)

    # 迭代
    # 找到重复的区间，删除重复的区间
    def deleteDuplicates1(self, head: ListNode) -> ListNode:

        dummy = ListNode()
        dummy.next = head
        pre = dummy
        cur = pre.next

        while (cur and cur.next):
            if cur.val == cur.next.val:
                cur = cur.next
            else:
                # 此时cur与next不同
                if pre.next is cur:
                    # 无重复
                    pre = cur
                    cur = cur.next
                else:
                    cur = cur.next
                    pre.next = cur
        # 最后重复处理，cur是最后一个重复的数，cur.next为NOne
        if pre.next is not cur:
            pre.next = None
        return dummy.next

# ok!
