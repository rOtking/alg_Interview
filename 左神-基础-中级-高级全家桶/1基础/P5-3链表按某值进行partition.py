'''
给定单链表与k，按链表小于k在左边，等于在中间，大于在右边，保持相对顺序不变。
'''
'''
初步：head1,head2,head3分别收集，将他们串起来。但是head1连head2是需要直到head1的尾结点，多一次遍历，还不如再加个tail结点来的方便。

最终:head1,tail1,head2,tail2,head3,tail3

注意：如果没有小于区呢？没有等于区呢？连None是不是有问题？

'''
# todo 链表的问题难的地方就是边界情况的处理！！！！

class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None

def listPartition(head, pivot):
    if head is None or head.next is None:
        return head
    head1, tail1, head2,tail2,head3,tail3 = None, None, None, None, None,None

    cur = head
    # 分别连接3条list
    while(cur is not None):
        # todo 先保存，再断开很关键！不然很容易循环起来的！
        next_ = cur.next
        cur.next = None
        if cur.val < pivot:
            # tail空就是第一个，首尾都是它
            if tail1 is None:
                head1 = cur
                tail1 = cur
            # tail不空，就在原来tail的基础上连起来，更新tail
            else:
                tail1.next = cur
                tail1 = cur

        elif cur.val == pivot:
            if tail2 is None:
                head2 = cur
                tail2 = cur
            else:
                tail2.next = cur
                tail2 = cur

        else:
            if tail3 is None:
                head3 = cur
                tail3 = cur
            else:
                tail3.next = cur
                tail3 = cur
        cur = next_

    # 首尾相连
    # todo 注意空的情况
    if head1 is None:
        if head2 is None:
            return head3
        else:
            tail2.next = head3
            return head2
    else:
        if head2 is None:
            tail1.next = head3
        else:
            tail1.next = head2
            tail2.next = head3
        return head1

def printList(head):
    cur = head
    while(cur):
        print(cur.val)
        cur = cur.next
# 情况1   都有
# 3,4,5,6,2,7,5     pivot = 5  ---->>3,4,2,5,5,6,7
head1 = ListNode(3)
p1 = ListNode(4)
p2 = ListNode(5)
p3 = ListNode(6)
p4 = ListNode(2)
p5 = ListNode(7)
p6 = ListNode(5)
head1.next = p1
p1.next = p2
p2.next = p3
p3.next = p4
p4.next = p5
p5.next = p6

# head = listPartition(head1, 5)
# printList(head)

# 情况2   缺head1与head2
# 7，4，3   p = 1
head2 = ListNode(7)
p6 = ListNode(4)
p7 = ListNode(3)
head2.next = p6
p6.next = p7


# todo ok 全篇没问题！
head = listPartition(head2, 1)
printList(head)


