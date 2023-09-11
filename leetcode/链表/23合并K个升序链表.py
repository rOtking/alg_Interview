# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
import heapq


class Solution:
    # def mergeKLists(self, lists: List[ListNode]) -> ListNode:
    #     arr = []

    #     for head in lists:
    #         cur = head
    #         while(cur != None):
    #             arr.append(cur.val)
    #             cur = cur.next

    #     arr = sorted(arr)
    #     if len(arr) == 0:
    #         return None
    #     head = ListNode(arr[0])
    #     pre = head

    #     for v in arr:
    #         cur = ListNode(v)
    #         pre.next = cur
    #         pre = cur

    #     return head.next

    # def mergeKLists(self, lists) -> ListNode:
        # if lists is None or len(lists) == 0:
        #     return None
        #
        # res = []
        # heapq.heapify(res)
        # for i, eachHead in enumerate(lists):
        #     cur = eachHead
        #     j = 0
        #     while (cur):
        #         heapq.heappush(res, (cur.val, i, j, cur))
        #         cur = cur.next
        #         j += 1
        #
        # dummy = ListNode()
        # cur = dummy
        # while (len(res) != 0):
        #     node = heapq.heappop(res)
        #     cur.next = node[3]
        #     cur = cur.next
        #
        # return dummy.next

    # 分治

    # L-R上合并
    def process(self, lists, L, R):
        if L == R:
            return lists[L]


        # todo 这里一定加优先级！！！！
        mid = L + ((R - L) >> 1)
        head1 = self.process(lists, L, mid)
        head2 = self.process(lists, mid + 1, R)

        head = self.merge(head1, head2)
        return head

    def merge(self, head1, head2):

        if head1 is None:
            return head2
        if head2 is None:
            return head1

        dummy = ListNode()
        cur = dummy
        while(head1 and head2):
            if head1.val < head2.val:
                cur.next = head1
                head1 = head1.next
            else:
                cur.next = head2
                head2 = head2.next
            cur = cur.next

        cur.next = head1 if head1 else head2
        return dummy.next

    def mergeKLists(self, lists) -> ListNode:
        if lists is None or len(lists) == 0:
            return None

        head = self.process(lists, 0, len(lists) - 1)
        return head




h1 = ListNode(-1)
p1 = ListNode(5)
p2 = ListNode(11)
h1.next = p1
p1.next = p2

h2 = ListNode(1)
p3 = ListNode(3)
p4 = ListNode(4)
h2.next = p3
p3.next = p4

h3 = ListNode(6)
h3.next = ListNode(10)

sol = Solution()
sol.mergeKLists([None, h1, None, h3])


'''
            方法很多，但是算法性不强
1.就是直接把node打散，放入arr中排序。
2.利用堆，k容量的堆能节省空间！
3.顺序合并，list1与list2合，依次与后面合。
4.分治，两两合并。（推荐使用，归并的拓展！！！）


注意：

heapy排序的原理是 < ，所以可以比较的对象才能用python自带的堆排序。

如果自定义类型，有两种加入heapy的方式

1.自定义比较函数；
2.组成一个元祖，tuple的比较是逐个元素依次比较。
'''
class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

    # 自定义比较函数
    def __lt__(self, other):
        return self.val < other.val


'''
这样 Node类型就可以加入heapy了。

res = []
heapy.heappush(res, (node.val, node))

直接包装tuple，node带着，利用tuple的可比较性！

'''
