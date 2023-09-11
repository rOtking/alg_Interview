# Definition for a Node.
class Node:
    def __init__(self, val, prev=None, next=None, child=None):
        self.val = val
        self.prev = prev
        self.next = next
        self.child = child



class Solution:
    def flatten(self, head: 'Node') -> 'Node':
        # todo 注意！
        if head is None or (head.next is None and head.child is None):
            return head

        # 递归
        if head.child is None:
            next_ = self.flatten(head.next)
            return head
        else:
            # 可能是None
            nextHead = self.flatten(head.next)
            childHead = self.flatten(head.child)
            # 找tail
            childTail = childHead
            while (childTail.next is not None):
                childTail = childTail.next
            head.next = childHead
            childHead.prev = head
            head.child = None
            childTail.next = nextHead
            if nextHead:
                nextHead.prev = childTail
            return head

# todo ok  但是重复看！（1）边界条件；（2）None类型的next与prev

p1 = Node(1)
p2 = Node(2)
p3 = Node(3)
p4 = Node(4)
p5 = Node(5)
p6 = Node(6)
p7 = Node(7)
p8 = Node(8)
p9 = Node(9)
p10 = Node(10)
p11 = Node(11)
p12 = Node(12)


p1.next = p2
p2.prev = p1
p2.next = p3

p3.prev = p2
p3.next = p4
p3.child = p7

p4.prev = p3
p4.next = p5

p5.prev = p4
p5.next = p6

p6.prev = p5

p7.next = p8
p8.prev = p7
p8.next = p9
p8.child = p11

p9.prev = p8
p9.next = p10

p11.next = p12
p12.prev = p11

p10.prev = p9

sol = Solution()
head = sol.flatten(p1)

while(head):
    print(head.val)
    head = head.next
