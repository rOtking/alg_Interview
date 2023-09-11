# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:

    def partition(self, head: ListNode, x: int) -> ListNode:
        if head is None or head.next is None:
            return head

        dummy1 = ListNode(0)
        dummy2 = ListNode(0)
        tail1 = None  # 链表1的尾，方便连接
        tail2 = None

        cur = head
        while (cur):
            # todo 尽量断开联系，以免循环链表！！！！
            next_ = cur.next
            cur.next = None
            if cur.val < x:
                if tail1 is None:
                    dummy1.next = cur
                    tail1 = cur
                else:
                    tail1.next = cur
                    tail1 = tail1.next
            else:
                if tail2 is None:
                    dummy2.next = cur
                    tail2 = cur
                else:
                    tail2.next = cur
                    tail2 = tail2.next
            cur = next_
        # 边界 NOne
        if tail1:
            tail1.next = dummy2.next
            return dummy1.next
        else:
            return dummy2.next

'''
ok的
'''


def printList(head):
    cur = head
    res = []
    while(cur is not None):
        res.append(cur.val)
        cur = cur.next
    print(res)

if __name__ == '__main__':
    a = ListNode(2)
    b = ListNode(1)



    a.next = b


    printList(a)

    s = Solution()
    head = s.partition(a, 2)
    printList(head)



# ok!
# 细节还是挺多的！注意一下吧
# 小于区的首尾，大于区的首尾，4个变量来记录，不然容易乱！


# todo 或者简单点，空间O(n)，拆分再合并
# todo 终于通过了，前面加一个哑节点瞬间好处理多了，避免了头节点插入与中间节点插入不同意的问题！Nice！








