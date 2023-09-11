# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    '''
    额外空间O(1)，后半逆序
    '''

    def reverseListNode(self, root):
        '''
        链表逆序函数
        '''
        pre = root
        if pre.next is None:
            return pre
        cur = pre.next
        pre.next = None
        while (cur is not None):
            post = cur.next
            cur.next = pre
            pre = cur
            cur = post

        head = pre
        return head

    def isPalindrome(self, head) -> bool:
        if head is None or head.next is None:
            return True
        # 找中点
        fast = head
        slow = head
        while (fast is not None and fast.next is not None):
            slow = slow.next
            fast = fast.next.next
        # 此时slow是中点
        end = self.reverseListNode(slow)  # 逆转后的尾节点，也是后半部分的头节点
        cur1 = head
        cur2 = end
        flag = True
        while (cur1 is not None and cur2 is not None):
            if cur1.val != cur2.val:
                flag = False
                break
            else:
                cur1 = cur1.next
                cur2 = cur2.next
                continue

        # 记得把链表调整回原来的顺序！
        head2 = self.reverseListNode(end)  # 第2段的head
        return flag

# ok
# 方法1：逆序，再比较 空间O(n)
# 方法2：递归的解决，其实就是递归解决逆序的过程，只不过需要一个全局的left指针！空间O(n)
# 方法3：快慢双指针逆序一半，空间O(1)
