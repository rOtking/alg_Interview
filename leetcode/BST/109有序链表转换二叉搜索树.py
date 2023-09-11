# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def sortedListToBST(self, head: ListNode) -> TreeNode:
        '''
        1.找中点；2.左右链表递归生成；3 连接
        :param head:
        :return:
        '''
        if head is None:
            return None
        if head.next is None:
            # todo 这里要返回BT 不是ListNode
            root = TreeNode(head.val)
            return root

        # 1. 中点 pre是中点前一个
        dummy_node = ListNode(0)
        dummy_node.next = head
        pre, slow, fast = dummy_node, head, head
        while(fast.next is not None and fast.next.next is not None):
            pre = pre.next
            slow = slow.next
            fast = fast.next.next

        pre.next = None

        root = TreeNode(slow.val)
        root.left = self.sortedListToBST(dummy_node.next)   # todo 精髓
        root.right = self.sortedListToBST(slow.next)
        slow.next = None
        dummy_node.next = None

        return root

if __name__ == '__main__':
    a = ListNode(-10)
    b = ListNode(-3)
    c = ListNode(0)
    d = ListNode(5)
    e = ListNode(9)

    a.next = b
    b.next = c
    c.next = d
    d.next = e

    s = Solution()
    root = s.sortedListToBST(a)
    print(root)


# ok
# 注意细节