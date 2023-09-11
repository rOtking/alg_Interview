# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:

    # todo 自己的做法是ok的，但是时间有点慢！
    def insertionSortList1(self, head: ListNode) -> ListNode:
        if head is None or head.next is None:
            return head

        dummy_node = ListNode(0)
        dummy_node.next = head
        cur = head.next
        head.next = None
        while(cur is not None):
            next_ = cur.next
            cur.next = None
            tmp = dummy_node.next
            # 寻找cur在已排序的list中的位置
            if cur.val < tmp.val:
                dummy_node.next = cur
                cur.next = tmp
            else:
                while(tmp.next is not None and tmp.next.val < cur.val):
                    tmp = tmp.next
                if tmp.next is None:
                    tmp.next = cur
                else:
                    tmp1 = tmp.next
                    tmp.next = cur
                    cur.next = tmp1

            cur = next_


        return dummy_node.next

    # 至少1个结点
    # todo end是已经排好的尾; cur是待插入的,cur=end.next；  pre<cur<pre.next则插入或cur本身就比end大，不变。
    def insertionSortList(self, head: ListNode) -> ListNode:
        dummy = ListNode(0)
        dummy.next = head
        end = head
        cur = end.next # 关键！！
        end.next = None
        while(cur):
            theNext = cur.next
            cur.next = None   # 先断开
            pre = dummy
            while(pre is not end and pre.next.val <= cur.val):
                pre = pre.next
            # pre一定比cur小；pre是end或者pre后一个比cur大
            if pre is end:
                end.next = cur
                end = end.next    # 因为cur本身就在end后面
            if pre.next.val > cur.val:
                cur.next = pre.next
                pre.next = cur
                # end不变
            cur = theNext
        return dummy.next








# todo 核心是：list是单向的，所以在插入的时候只能从前向后找。

# todo 链表因为交换比较复杂，插入比较方便，所以适合使用移动+插入的实现方式。

if __name__ == '__main__':
    p1 = ListNode(4)
    p2 = ListNode(2)
    p3 = ListNode(1)
    p4 = ListNode(3)

    p1.next = p2
    p2.next = p3
    p3.next = p4

    s = Solution()
    node = s.insertionSortList(p1)
    while(node):
        print(node.val)