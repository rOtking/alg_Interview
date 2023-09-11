
class ListNode():
    def __init__(self, val=0, next=None):
        super(ListNode, self).__init__()
        self.val = val
        self.next = next

    def traverse(self):
        res = []
        node = self
        while(node != None):
            res.append(node.val)
            node = node.next
        print(res)

class List():
    def __init__(self, arr):
        super(List, self).__init__()
        self.head = self._bulid(arr)

    def _bulid(self, arr):
        head = ListNode()
        pre = head
        for ele in arr:
            cur = ListNode(ele)
            pre.next = cur
            pre = cur
        return head.next

    def vaule_print(self):
        cur = self.head
        res = []
        while(cur != None):
            res.append(cur.val)
            cur = cur.next

        print(res)



if __name__ == '__main__':
    l = List([1,2,3,4])
    l.vaule_print()



