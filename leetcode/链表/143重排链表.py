# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def reorderList(self, head: ListNode) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        if head is None or head.next is None:
            return head
        # 按遍历顺序存节点
        help = []
        cur = head
        while(cur is not None):
            help.append(cur)
            pre = cur
            cur = cur.next
            pre.next = None
        cur = head
        start = 1
        end = len(help) - 1
        while(start < end):
            cur.next = help[end]
            cur = help[end]
            cur.next = help[start]
            cur = help[start]
            start += 1
            end -= 1
        if start == end:
            cur.next = help[start]
            help[start].next = None

        return head

def printList(head):
    cur = head
    res = []
    while(cur is not None):
        res.append(cur.val)
        cur = cur.next
    print(res)

if __name__ == '__main__':
    a = ListNode(1)
    b = ListNode(2)
    c = ListNode(3)
    d = ListNode(4)
    e = ListNode(5)

    a.next = b
    b.next = c
    c.next = d
    d.next = e

    printList(a)
    s = Solution()
    head = s.reorderList(a)
    printList(head)


# ok
# 就是按题意连接，没啥难度。

'''
            有点东西
（1）丢进arr中，通过arr下标可O(1)获取实现重连，但是空间O(n)
（2）找中点，后半逆序，双指针外排。空间O(1)

List的逆序访问甚至任意位置访问都可以用arr辅助数组来方便实现，只是空间O(n)，不要求空间这个方法就太牛逼了！！！！


'''




