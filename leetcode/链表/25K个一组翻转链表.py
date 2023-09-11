# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        if head is None:
            return head

        cur = head
        for _ in range(k - 1):
            if cur is None:
                break
            cur  = cur.next

        # 要么cur是None，要么cur是第k个元素
        if cur is None:
            return head
        else:
            next_head = cur.next
            cur.next = None
            new_head = self.reverseRecursion(head)
            next_new_head = self.reverseKGroup(next_head, k)
            head.next = next_new_head
            return new_head




    def reverseIteration(self, head):
        '''
        迭代的反转list
        :return:
        '''
        if head is None or head.next is None:
            return head

        pre = None
        cur = head
        next_ = head.next

        while(next_ is not None):
            cur.next = pre
            pre = cur
            cur = next_
            next_ = next_.next

        # 跳出时next_就是None，cur就是新的head，但是要连一下
        cur.next = pre
        return cur


    def reverseRecursion(self, head):
        '''
        递归的反转list
        :return:
        '''
        if head.next is None:
            return head

        new_head = self.reverseRecursion(head.next)
        new_last = head.next
        head.next = None
        new_last.next = head
        return new_head


# ok
# 没问题，把问题拆解开就很简单了！
# 只要是反转list延伸出来的问题，基本上都用到了反转链表的函数作为其中的工具。

# 两个版本的反转都是正确的！nb啊 哈哈哈
