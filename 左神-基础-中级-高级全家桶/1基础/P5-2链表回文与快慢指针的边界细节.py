'''
基础的单链表，双向链表不说了。
'''
'''
链表问题基本都是纯coding能力，能用到算法的很少！这里列几个需要会的题，不写代码了解释了。

有几个用到算法的，写出来
'''
'''
1.反转单链表与双向链表。------
2.打印两个有序链表公共部分。------双指针，谁小谁移动

重要技巧：
（1）额外的存储结果，如哈希表
（2）快慢指针
'''



# ------------------- 1.判断链表回文 ----------------#
'''
要求时间O(n)，空间O(1)

不要求空间用stack即可。

关键：通过快慢指针找中点，并将中点之后的部分进行逆序！ 
'''
# todo 已验证 ok的！
def isPalindrome(self, head) -> bool:
    # 1.找中点
    if head is None and head.next is None:
        return True
    # 至少有一个结点
    slow, fast = head, head
    while (fast is not None and fast.next is not None):
        slow = slow.next
        fast = fast.next.next
    # 此时slow就是中点,开始逆序
    pre = None
    cur = slow
    while (cur is not None):
        next_ = cur.next
        cur.next = pre
        pre = cur
        cur = next_
    # 定义首尾结点
    p1 = head
    p2 = pre
    while (p1 is not None and p2 is not None):
        if p1.val != p2.val:
            return False
        p1 = p1.next
        p2 = p2.next

    # todo list恢复原来的顺序！
    return True

# ------------------------------------------------#

# todo 快慢指针的边界细节
# ------------------- 2.快慢指针细节coding ----------------#

'''
（1）找中点！
fast走完时，slow在中点。
1,2,3,2,1 模式下：找到3
1,2,2,1   模式下：找到第一个2
尤其链表长度为1个，2个，3个这种小数据的时候也要有效。

'''
class ListNode:
    def __init__(self, value):
        self.value = value
        self.next = None
head1 = ListNode(1)
p1 = ListNode(2)
p2 = ListNode(3)
p3 = ListNode(2)
p4 = ListNode(1)
head1.next = p1
p1.next = p2
p2.next = p3
p3.next = p4
# head1 为 1，2，3，2，1

head2 = ListNode(1)
p5 = ListNode(2)
p6 = ListNode(2)
p7 = ListNode(1)
head2.next = p5
p5.next = p6
p6.next = p7
# head2   1，2，2，1

head3 = ListNode(1)
p8 = ListNode(2)
p9 = ListNode(1)
head3.next = p8
p8.next = p9
# head3   1,2,1

head4 = ListNode(1)
p10 = ListNode(1)
head4.next = p10
# head4   1,1

head5 = ListNode(1)



def findMid(head):
    slow, fast = head, head
    if fast is None:
        return None
    while(fast.next is not None and fast.next.next is not None):
        slow = slow.next
        fast = fast.next.next

    # 只是验证，打印中点之后的部分
    # while(slow is not None):
    #     print(slow.value)
    #     slow = slow.next

    return slow

# 实验 正确
# _ = findMid(head5)

'''
（2）找中点前一个

模式：1，2，3，2，1   找到第一个2
模式：1，2，2，1     找到第一个1
    1，3，1       第一个1
    1，1         None
    1           None

fast先走一步，每次两步，验证三步
'''
def findMidPre(head):

    fast, slow  = head, head
    if fast is None or fast.next is None or fast.next.next is None:
        return None
    # fast.next.next存在，即至少3个数
    fast = fast.next
    while(fast.next is not None and fast.next.next is not None and fast.next.next.next is not None):
        slow = slow.next
        fast = fast.next.next
    # 只是验证，打印中点之后的部分
    # while(slow is not None):
    #     print(slow.value)
    #     slow = slow.next


# 实验 正确
# _ = findMidPre(head3)


'''
（3）找倒数第n个
模式：1，2，3，2，1  倒数第1是1，倒数第6是None

1，2 倒数1是1

1   倒数1是1，倒数2是None

fast先走n-1步，fast走到最后一个元素，slow就是倒数第n个
'''
def findLastN1(head, n):
    if head is None:
        return None

    fast, slow = head, head
    for _ in range(n - 1):
        fast = fast.next
        # todo 顺序很重要
        if fast is None:
            return None
    # fast至少是尾结点
    while(fast.next is not None):
        fast = fast.next
        slow = slow.next
    # 只是验证，打印中点之后的部分
    while(slow is not None):
        print(slow.value)
        slow = slow.next
    return slow

# todo 更好一点
def findLastN(head, n):
    dummy = ListNode(0)
    dummy.next = head
    fast, slow = dummy, dummy
    for _ in range(n):
        fast = fast.next
        if fast is None:
            return None
    # todo 如果是找倒数n+1，就是     while(fast.next)   !!!!
    while(fast):
        fast = fast.next
        slow = slow.next
    while(slow is not None):
        print(slow.value)
        slow = slow.next
    return slow


# ok
_ = findLastN(head5, 1)

'''
（3）找倒数第n+1，以便进行第n个的删除操作
分两种情况：<1>存在倒数第n+1个，找到即可；<2>倒数第n个是head，直接去掉。

'''

def removeNthFromEnd(head, n):
    if head is None:
        return None

    slow, fast = head, head
    for _ in range(n):
        fast = fast.next
        if fast is None:
            if _ == n - 1:    # todo 删除的是head 单独处理！
                return head.next
            else:
                return None
    # fast至少是尾结点
    while (fast.next is not None):
        slow = slow.next
        fast = fast.next
    removeNode = slow.next
    next_ = removeNode.next
    slow.next = next_
    removeNode.next = None
    del removeNode
    return head
# ------------------------------------------------#
