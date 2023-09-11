# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    # def sortedListToBST(self, head: ListNode) -> TreeNode:
        # '''
        # 1.找中点；2.左右链表递归生成；3 连接
        # :param head:
        # :return:
        # '''
        # if head is None:
        #     # 这里要返回BT 不是ListNode
        #     return None
        # if head.next is None:
        #     root = TreeNode(head.val)
        #     return root

        # # 1. 中点 pre是中点前一个
        # dummy_node = ListNode(0)
        # dummy_node.next = head
        # pre, slow, fast = dummy_node, head, head
        # while(fast.next is not None and fast.next.next is not None):
        #     pre = pre.next
        #     slow = slow.next
        #     fast = fast.next.next

        # pre.next = None

        # root = TreeNode(slow.val)
        # root.left = self.sortedListToBST(dummy_node.next)   # todo 精髓
        # root.right = self.sortedListToBST(slow.next)
        # slow.next = None
        # dummy_node.next = None

        # return root


    def sortedListToBST(self, head: ListNode) -> TreeNode:
        if head is None:
            return head
        if head.next is None:
            return TreeNode(head.val)

        # todo 找中点前一个是个套路，要记住，现推太难受了
        slow, fast = head, head.next
        while(fast.next and fast.next.next and fast.next.next.next):
            slow = slow.next
            fast = fast.next.next
        mid = slow.next
        # todo # 断开链接很关键  不然会重复调用！
        slow.next = None
        leftRoot = head
        rightRoot = mid.next
        root = TreeNode(mid.val)
        root.left = self.sortedListToBST(leftRoot)
        root.right = self.sortedListToBST(rightRoot)

        return root














