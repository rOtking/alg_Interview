# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

'''
设环前距离m,环长n，快慢指针相遇点离入环位置k
则 2(m + k) = m + n + k
m = n - k
也就是两个指针，指针1从头走，指针2从交点走，相遇就是入环位置

'''
class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        if head is None or head.next is None:
            return None
        slow, fast = head, head
        while(fast and fast.next):
            slow = slow.next
            fast = fast.next.next
            if slow is fast:
                break
        # 无环
        if fast is None or fast.next is None:
            return None

        # (a + b)*2 = a+m+b.  m = a+b. 求a=m-b
        p1 = head
        p2 = slow
        while(p1 is not p2):
            p1 = p1.next
            p2 = p2.next
        return p1

def printList(head):
    cur = head
    res = []
    while(cur is not None):
        res.append(cur.val)
        cur = cur.next
    print(res)


if __name__ == '__main__':
    a = ListNode(3)
    b = ListNode(2)
    c = ListNode(0)
    d = ListNode(-4)



    a.next = b
    b.next = c
    c.next = d
    d.next = b



    s = Solution()
    head = s.detectCycle(a)


# ok
# todo 注意 最开始fast = slow，这次不要进行判断！其他没啥

