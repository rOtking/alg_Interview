'''
https://zhuanlan.zhihu.com/p/386929820
'''
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def __init__(self):
        self.res = float('-inf')
    # -------------------------------- 链表9题 ----------------------------------- #

    # 15.删除链表的倒数第 N 个结点:https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/
    # list >= 1, n>=1
    # todo dummy,双指针，找倒数n+1，fast先走n
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        dummy = ListNode(0)
        dummy.next = head
        slow, fast = dummy, dummy
        for _ in range(n):
            fast = fast.next
        # fast至少是尾结点
        while(fast.next is not None):
            fast = fast.next
            slow = slow.next
        # slow是pre
        pre = slow
        toDel = pre.next
        theNext = toDel.next
        pre.next = theNext
        toDel.next = None
        del toDel
        return dummy.next

    # 17.合并两个有序链表:https://leetcode-cn.com/problems/merge-two-sorted-lists/
    # 可能是None，没有节点
    # todo 双指针，归并，设个pre记录当前的结尾即可
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if l1 is None:
            return l2
        if l2 is None:
            return l1
        # 都不空
        dummy = ListNode(0)
        pre = dummy
        p1, p2 = l1, l2
        while(p1 and p2):
            if p1.val < p2.val:
                pre.next = p1
                p1 = p1.next
            else:
                pre.next = p2
                p2 = p2.next
            pre = pre.next
        if p1:
            pre.next = p1
        else:
            pre.next = p2
        return dummy.next

    # 19.合并K个升序链表:https://leetcode-cn.com/problems/merge-k-sorted-lists/
    # todo 归并的两两合并
    def mergeKLists(self, lists) -> ListNode:
        # 吃一个list，返回一个node
        def process(lists, left, right):
            if left == right:
                return lists[left]
            mid = left + ((right - left) >> 1)
            head1 = process(lists, left, mid)
            head2 = process(lists, mid + 1, right)
            res = merge(head1, head2)
            return res
        def merge(head1, head2):
            p1, p2 = head1, head2
            dummy = ListNode(0)
            pre = dummy
            while(p1 and p2):
                if p1.val <= p2.val:
                    pre.next = p1
                    p1 = p1.next
                else:
                    pre.next = p2
                    p2 = p2.next
                pre = pre.next
            pre.next = p1 if p1 else p2
            return dummy.next
        if len(lists) == 0:
            return None
        res = process(lists, 0, len(lists) - 1)
        return res


    # 71.复制带随机指针的链表:https://leetcode-cn.com/problems/copy-list-with-random-pointer/
    # 可能 None
    # todo hash：两次遍历，第一次连next并存old-new的查找；第二次遍历next时连random，同的old.random找到new.random，连上
    def copyRandomList1(self, head: 'Node') -> 'Node':
        if head is None:
            return None
        map1 = {}
        newHead = Node(head.val)

        map1[head] = newHead
        cur, newCur = head, newHead
        # 连next
        while(cur.next):
            cur = cur.next
            newCur = newCur.next
            newCur = Node(cur.val)
            map1[cur] = newCur

        # 连random
        cur, newCur = head, newHead
        while(cur):
            theRandom1 = cur.random
            newRandom = map1[theRandom1] if theRandom1 else None
            newCur.random = newRandom
            cur = cur.next
            newCur = newCur.next

        return newHead

    # todo node1-node1'-node2-node2'    第一次遍历:复制。第二次，node'.random = node.random.next连random。第三次：分离！
    def copyRandomList(self, head: 'Node') -> 'Node':
        if head is None:
            return None
        cur = head
        # 复制结点
        while(cur):
            theNext = cur.next
            newNode = Node(cur.val)
            cur.next = newNode
            newNode.next = theNext
            cur = theNext
        cur = head
        # 连接random
        while(cur):
            newNode = cur.next
            newRandom = cur.random.next if cur.random else None
            newNode.random = newRandom
            cur = cur.next.next
        # 分离
        newHead = head.next
        pre1, pre2 = head, newHead
        while(pre2.next):
            pre1.next = pre1.next.next
            pre2.next = pre2.next.next
            pre1 = pre1.next
            pre2 = pre2.next

        return newHead

    # 75.排序链表:https://leetcode-cn.com/problems/sort-list/
    # 可能为空
    # todo 归并找中点 为了保险，中点与next断开
    def sortList(self, head: ListNode) -> ListNode:
        def process(head):
            if head is None or head.next is None:
                return head
            # 至少1个node 找中点
            fast, slow = head, head
            while(fast.next and fast.next.next):
                slow = slow.next
                fast = fast.next.next
            new_head = slow.next
            slow.next = None
            left_head = process(head)
            right_head = process(new_head)
            return merge(left_head, right_head)

        def merge(left_head, right_head):
            if left_head is None:
                return right_head
            if right_head is None:
                return left_head
            p1, p2 = left_head, right_head
            dummy = ListNode(0)
            pre = dummy
            while(p1 and p2):
                if p1.val <= p2.val:
                    pre.next = p1
                    p1 = p1.next
                else:
                    pre.next = p2
                    p2 = p2.next
                pre = pre.next
            pre.next = p1 if p1 else p2
            return dummy.next
        res = process(head)
        return res

    # 84.相交链表:https://leetcode-cn.com/problems/intersection-of-two-linked-lists/
    # 大于等于1个元素
    # todo 先走距离差，再一次判断   O(max(m,n))
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        # A的长度
        longA, longB = 0, 0
        pA, pB = headA, headB
        while(pA):
            pA = pA.next
            longA += 1
        while(pB):
            pB = pB.next
            longB += 1
        # fast是先走的
        if longA == 0 or longB == 0:
            return None
        # todo 重新赋值，不能用 pA pB 都已经走到头了，都是None
        fast, slow = (headA, headB) if longA > longB else (headB, headA)
        diff = longA - longB if longA - longB > 0 else longB - longA
        for _ in range(diff):
            fast = fast.next

        while(fast and fast is not slow):
            fast = fast.next
            slow = slow.next
        # 不相交就是None，相交就是交点，都是fast
        return fast

    # 96.反转链表:https://leetcode-cn.com/problems/reverse-linked-list/
    # 个数 [0,5000]
    # todo pre是已反转完的head cur是待反转的head
    def reverseList(self, head: ListNode) -> ListNode:
        pre = None
        cur = head
        while(cur):
            theNext = cur.next
            cur.next = pre
            pre = cur
            cur = theNext
        return pre

    # 104.回文链表:https://leetcode-cn.com/problems/palindrome-linked-list/
    # 至少一个数
    # 1->2->3->2->1   找到3逆序   1->2->3<-2<-1
    # 1->2->2->1  找到第一个2逆序  1->2<-2<-1
    # todo 找中点 + 后半逆序
    def isPalindrome(self, head) -> bool:
        if head is None or head.next is None:
            return True
        # 至少两个元素
        slow, fast = head, head
        while(fast and fast.next and fast.next.next):
            slow = slow.next
            fast = fast.next.next

        # slow是中点
        head2 = self.reverseList(slow)
        p1, p2 = head, head2
        res = True
        while(p1 and p2):
            if p1.val != p2.val:
                res = False
                break
            p1 = p1.next
            p2 = p2.next
        # 顺序调回来
        _ = self.reverseList(head2)
        return res
    # todo 借助O(n)空间的方法
    def isPalindrome1(self, head) -> bool:
        help = []
        cur = head
        while(cur):
            help.append(cur.val)
            cur = cur.next

        i, j = 0, len(help) - 1
        while(i < j and help[i] == help[j]):
            i += 1
            j -= 1
        return False if i < j else True

    # 106.删除链表中的结点:https://leetcode-cn.com/problems/delete-node-in-a-linked-list/
    # node有效，不是tail
    # todo 把自己复制成next，就有了pre
    def deleteNode(self, node):
        cur = node.next  # 可能是尾巴
        theNext = cur.next
        node.val = cur.val
        node.next = theNext

    # 74.LRU缓存机制:https://leetcode-cn.com/problems/lru-cache/
    # todo 先不看了！！
    # ---------------------------------------------------------------------------- #

    # -------------------------------- 二叉树12题 ----------------------------------- #

    # 50.二叉树的中序遍历:https://leetcode-cn.com/problems/binary-tree-inorder-traversal/
    # todo 递归
    def inorderTraversal1(self, root: TreeNode) -> List[int]:
        res = []
        if root is None:
            return res
        left = self.inorderTraversal1(root.left)
        right = self.inorderTraversal1(root.right)
        res.extend(left)
        res.append(root.val)
        res.extend(right)
        return res
    # todo 非递归
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        stack = []
        cur = root
        while(cur or len(stack) != 0):
            # 有左
            if cur:
                stack.append(cur)
                cur = cur.left
            # 无左
            else:
                cur = stack.pop()
                res.append(cur.val)
                cur = cur.right
        return res
    # 52.验证二叉搜索树:https://leetcode-cn.com/problems/validate-binary-search-tree/submissions/
    # 至少一个结点
    # todo 树形DP + 返回辅助信息 + 停止条件是空结点与叶子结点！！！
    def isValidBST1(self, root: TreeNode) -> bool:
        def process(root):
            if root is None:
                return None
            if root.left is None and root.right is None:
                return True, root.val, root.val
            res1 = process(root.left)
            res2 = process(root.right)
            isValid1 = True if (res1 is None) or (res1[0] and root.val > res1[2]) else False
            isValid2 = True if (res2 is None) or (res2[0] and root.val < res2[1]) else False

            # todo 返回 [是否BST，最小值，最大值]
            return isValid1 and isValid2, res1[1] if res1 else root.val, res2[2] if res2 else root.val
        return process(root)[0]
    # todo 递归中序 + 升序
    def isValidBST2(self, root: TreeNode) -> bool:
        def process(root):
            res = []
            if root is None:
                return res
            res1 = process(root.left)
            res2 = process(root.right)
            res.extend(res1)
            res.append(root.val)
            res.extend(res2)
            return res

        res = process(root)
        for i in range(len(res) - 1):
            if res[i] >= res[i + 1]:
                return False
        return True
    # todo 迭代中序 + 升序
    def isValidBST3(self, root: TreeNode) -> bool:
        if root is None:
            return False
        max_value = float('-inf')
        stack = []
        cur = root
        while(len(stack) != 0 or cur):
            if cur:
                stack.append(cur)
                cur = cur.left
            else:
                # todo 第二次访问时输出：中序！
                cur = stack.pop()
                if cur.val > max_value:
                    max_value = cur.val
                    cur = cur.right
                else:
                    return False
        return True

    # 53.对称二叉树:https://leetcode-cn.com/problems/symmetric-tree/
    # 至少一个node
    # todo BFS + 为空时占位符的处理
    def isSymmetric1(self, root: TreeNode) -> bool:
        queue = [root]
        while(len(queue) != 0):
            queue_size = len(queue)
            tmp = []
            for _ in range(queue_size):
                cur = queue.pop(0)
                if cur:
                    tmp.append(cur.val)
                else:
                    tmp.append('#')   # 为空的占位符
                if cur:
                    queue.append(cur.left)
                    queue.append(cur.right)
            i, j = 0, len(tmp) - 1
            while(i <= j):
                if tmp[i] == tmp[j]:
                    i += 1
                    j -= 1
                else:
                    return False
        return True

    # todo 直接不能树形DP，转为两个树的比较 + 判断两个树p与q是不是对称（注意是对称！不是相同！
    # todo 这个代码比BFS代码简单！！！！
    def isSymmetric2(self, root: TreeNode) -> bool:
        def check(p, q):
            if p is None and q is None:
                return True
            if p is None or q is None:
                return False

            res1 = check(p.left, q.right)
            res2 = check(p.right, q.left)
            res3 = p.val == q.val
            return res1 and res2 and res3
        return check(root.left, root.right)

    # 54.二叉树的层序遍历:https://leetcode-cn.com/problems/binary-tree-level-order-traversal/
    # left与right分别递归，拼结果不说了
    # todo BFS
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        res = []
        if root is None:
            return res
        queue = [root]
        while(len(queue) != 0):
            tmp = []
            queue_size = len(queue)
            for _ in range(queue_size):
                cur = queue.pop(0)
                tmp.append(cur.val)
                if cur.left:
                    queue.append(cur.left)
                if cur.right:
                    queue.append(cur.right)
            res.append(tmp)
        return res

    # 55.二叉树的锯齿形层序遍历:https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/submissions/
    # 可能为空
    # todo 直接BFS！！树形DP拼结果太麻烦了！
    # todo 始终是 从左到右存，注意输出的位置pop()还是pop(0)，append还是insert(0)!
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        res =[]
        if root is None:
            return res
        queue = [root]
        left2right = True  # 本轮 是否从左到右
        while(len(queue) != 0):
            size = len(queue)
            tmp = []
            for _ in range(size):
                if left2right:
                    # 现在queue是 左-右 存的
                    cur = queue.pop(0)
                    tmp.append(cur.val)
                    if cur.left:
                        queue.append(cur.left)
                    if cur.right:
                        queue.append(cur.right)
                else:
                    cur = queue.pop()
                    tmp.append(cur.val)
                    if cur.right:
                        queue.insert(0, cur.right)
                    if cur.left:
                        queue.insert(0, cur.left)
            left2right = not left2right
            res.append(tmp)
        return res

    # 56.二叉树的最大深度:https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/
    # todo 二叉树的树形DP就是去掉冗余信息的DFS！！
    # todo 可以BFS数层数，复杂了没必要！
    def maxDepth(self, root: TreeNode) -> int:
        if root is None:
            return 0
        if root.left is None and root.right is None:
            return 1

        left_depth = self.maxDepth(root.left)
        right_depth = self.maxDepth(root.right)
        return max(left_depth, right_depth) + 1

    # 57.从前序与中序遍历序列构造二叉树:https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/
    # 至少一个,且无重复元素
    # todo 根据root在前序与中序的位置，递归的构造左右子树
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        if len(preorder) == 0:
            return None
        root = TreeNode(preorder[0])
        root_val = preorder[0]
        inorder_root_index = inorder.index(root_val)
        left = self.buildTree(preorder[1:inorder_root_index + 1], inorder[:inorder_root_index])
        right = self.buildTree(preorder[inorder_root_index + 1:], inorder[inorder_root_index + 1:])
        root.left = left
        root.right = right
        return root


    # 58.将有序数组转换为二叉搜索树:https://leetcode-cn.com/problems/convert-sorted-array-to-binary-search-tree/
    # 至少一个node
    # todo 取中点，二分的构造
    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        if len(nums) == 0:
            return None
        if len(nums) == 1:
            return TreeNode(nums[0])
        l, r = 0, len(nums) - 1
        mid = l + ((r - l) >> 1)
        root = TreeNode(nums[mid])
        left = self.sortedArrayToBST(nums[:mid])
        right = self.sortedArrayToBST(nums[mid + 1:])
        root.left = left
        root.right = right
        return root

    # 63.二叉树的最大路径和:https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/
    # todo 很难！记忆
    # todo 已root为根的树上的最大和 + 已root为一端的单链最大和
    # todo 左边为了保存访问过的子树上的最大；右边为了向上传递可能组成最大值的子拼图
    # todo 难点在于想清楚：上述两个是否能涵盖所有情况，以及辅助函数怎么写！
    def maxPathSum(self, root: TreeNode) -> int:
        # 更新子树上的最大，返回最大单链
        def subMax(root):
            if root is None:
                return 0
            left = max(subMax(root.left), 0)
            right = max(subMax(root.right), 0)
            self.res = max(root.val + left + right, self.res)
            return root.val + max(left, right)

        subMax(root)

        return self.res

    # 103.二叉搜索树中第K小的元素:https://leetcode-cn.com/problems/kth-smallest-element-in-a-bst/
    # todo BST绕不开的迭代中序
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        stack = []
        cur = root
        idx = 0
        while(len(stack) != 0 or cur):
            if cur:
                stack.append(cur)   # todo 打磨细节！不是cur.left
                cur = cur.left
            else:
                cur = stack.pop()
                idx += 1
                if idx == k:
                    return cur.val
                cur = cur.right

    # 105.二叉树的最近公共祖先:https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/
    # 至少2个结点
    # todo 分别存从root-p与root-q的路径
    def lowestCommonAncestor1(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        def findPath(root, node):
            if root is None:
                return []
            if root is node:
                return [node]
            left = findPath(root.left, node)
            right = findPath(root.right, node)
            if len(left) != 0:
                left.insert(0, root)
                return left
            if len(right) != 0:
                right.insert(0, root)
                return right
            return []
        p_path = findPath(root, p)
        q_path = findPath(root, q)
        i = 1
        while(i <= len(p_path) - 1 and i <= len(q_path) - 1 and p_path[i] == q_path[i]):
            i += 1
        if i <= len(p_path) - 1:
            return p_path[i - 1]
        else:
            return q_path[i - 1]

    # todo p与q互为父子，直接返回父；p与q不互为父子，向上汇聚
    # todo 记忆
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if root is None or root is p or root is q:
            return root

        # 空就是不在子树；不空就是找到
        res1 = self.lowestCommonAncestor(root.left, p, q)
        res2 = self.lowestCommonAncestor(root.right, p, q)

        if res1 and res2:
            return root

        return res1 if res1 else res2

    # 102.二叉树中所有距离为K的结点:https://leetcode-cn.com/problems/all-nodes-distance-k-in-binary-tree/solution/er-cha-shu-zhong-suo-you-ju-chi-wei-k-de-qbla/
    # todo 不会，先放弃了；DFS+map
    def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
        pass
    # ---------------------------------------------------------------------------- #


    # -------------------------------- 其他 ----------------------------------- #

    # 1.两数之和: https://leetcode-cn.com/problems/two-sum/
    # python本身不能返回排序前的索引，np可以；
    # todo hashmap是所有前面的值，后面的会匹配前面，所以不会丢！    一开始没想出来！！
    def twoSum(self, nums, target):
        hashmap = {}     # k=数值，v=nums中的位置
        for idx, value in enumerate(nums):
            need = target - value
            if need in hashmap:
                return [hashmap[need], idx]
            else:
                hashmap[value] = idx

    # 2.两数相加:https://leetcode-cn.com/problems/add-two-numbers/
    # todo 对 add_bit 归0与结尾的处理
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        p1, p2 = l1, l2
        add_bit = 0
        dummy = ListNode(0)
        pre = dummy
        while(p1 is not None and p2 is not None):
            cur = p1.val + p2.val + add_bit
            if cur > 9:
                add_bit = 1
                cur -= 10
            else:
                add_bit = 0
            node = ListNode(cur)
            pre.next = node
            p1 = p1.next
            p2 = p2.next
            pre = pre.next

        while(p1 is not None):
            cur = p1.val + add_bit
            if cur > 9:
                add_bit = 1
                cur -= 10
            else:
                add_bit = 0
            node =  ListNode(cur)
            pre.next = node
            p1 = p1.next
            pre = pre.next
        while(p2 is not None):
            cur = p2.val + add_bit
            if cur > 9:
                add_bit = 1
                cur -= 10
            else:
                add_bit = 0
            node =  ListNode(cur)
            pre.next = node
            p2 = p2.next
            pre = pre.next

        if add_bit != 0:
            node = ListNode(add_bit)
            pre.next = node

        return dummy.next

    # 3.无重复字符最长子串:https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/
    def lengthOfLongestSubstring(self, s: str) -> int:
        if len(s) == 0:
            return 0
        max_num = 1
        window = set()
        i, j = 0, 1
        while(j < len(s)):
            pass

        return max_num

    # 6.整数反转:自己定义   范围 INT_MIN, INT_MAX = -2**31, 2**31 - 1

    # 9.盛最多水的容器:首尾双指针，一边收缩一边更新即可。
    # 12.最长公共前缀:横向比较，两个and 为True就继续和后面的and，False直接break

    # 23.旋转图像（90度）:先对角线两两换，完事左右交换
    # 37。合并区间：先排序，再看首尾合并

    # 48.合并两个有序数组:num1有空位0，in-placce的合并，从后向前！

    # 88.轮转数组：三次翻转
    # todo 原始数组	                                 1 2 3 4 5 6 7
    #  翻转所有元素                                    7 6 5 4 3 2 1
    #  翻转 [0, k%(n-1)]如k=3,n=7,[0,3]               5 6 7 4 3 2 1
    #  再反转剩下的                                    5 6 7 1 2 3 4


    # 4.移动零:https://leetcode-cn.com/problems/move-zeroes/
    # todo 注意是保证非零元素的顺序
    # todo 直接冒泡超时！
    # todo slow是在找第一个0,fast是在slow后面的第一个非0,[slow, fast - 1]都是0
    # todo 主要是靠fast找非0来实现：[slow, fast - 1]都是0
    def moveZeroes(self, nums) -> None:
        slow, fast = 0, 0
        while(fast < len(nums)):
            if nums[fast] != 0:
                nums[slow], nums[fast] = nums[fast], nums[slow]
                slow += 1
            fast += 1

        return

    # 69.加油站：贪心
    # 4.寻找两个正序数组的中位数：二分找第k个数，难，不看了
    # 66.最长连续序列:用set来查
    # 13.三数之和:排序+双指针  难 不看

    # ---------------------------------------------------------------------------- #

    # -------------------------------- DFS回溯 ----------------------------------- #
    # todo 生成所有组合！就是DFS的标志！
    # 14.电话号码的字母组合:https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/
    # todo list里面改，外边也会变！！
    def letterCombinations(self, digits: str):
        # 通常candidates是全量数据，配合另一个指代层数的参数begin来确定当前层的 候选人
        # track是做过的选择
        def dfs(res, track, candidates, begin):
            if len(track) == len(candidates):
                res.append(track)
                return res
            # 本轮候选人
            thisCandidates = candidates[begin]
            for ch in thisCandidates:
                track += ch
                dfs(res, track, candidates, begin + 1)
                track = track[:-1]
            return

        res = []
        if len(digits) == 0:
            return res
        digit2phone = {
            '2': 'abc',
            '3': 'def',
            '4': 'ghi',
            '5': 'jkl',
            '6': 'mno',
            '7': 'pqrs',
            '8': 'tuv',
            '9': 'wxyz'
        }
        res = []
        phones = []
        for digit in digits:
            phones.append(digit2phone[digit])
        track = ''
        dfs(res, track=track, candidates=phones, begin=0)
        return res


    # 18.括号生成:https://leetcode-cn.com/problems/generate-parentheses/
    # todo 规则 '('数量不满就能放，')'需要track中'('严格大于')'才能方，不然就是无效的！
    #  传p与q表示track中的'(' ')'数量，不用calNum了---->已经优化了！
    def generateParenthesis(self, n: int):
        def dfs(self, track, n, leftNum, rightNum):
            # leftNum, rightNum分别是左右括号已经用了多少
            if len(track) == 2 * n:
                self.res.append(track)
                return
            candidates = []
            if leftNum < n:
                candidates.append('(')
            if leftNum > rightNum and rightNum < n:
                candidates.append(')')

            for candidate in candidates:
                track += candidate
                if candidate == '(':
                    self.dfs(track, n, leftNum + 1, rightNum)
                else:
                    self.dfs(track, n, leftNum, rightNum + 1)
                track = track[:-1]
            return

        res = []
        track = ''
        dfs(res, track, n)
        return res


    # 31.全排列:https://leetcode-cn.com/problems/permutations/
    # 已知不重复，排位置即可 deepcopy
    def permute(self, nums: List[int]) -> List[List[int]]:
        # 已做过选择的位置
        def dfs(res, track, candidates, existIndexs):
            if len(track) == len(candidates):
                # todo deepcopy
                res.append(track[:])
                return
            thisIndexs = []
            for idx, x in enumerate(candidates):
                if idx not in existIndexs:
                    thisIndexs.append(idx)
            for thisIndex in thisIndexs:
                track.append(candidates[thisIndex])
                existIndexs.append(thisIndex)
                dfs(res, track, candidates, existIndexs)
                track.pop()
                existIndexs.pop()
            return
        res = []
        dfs(res, track=[], candidates=nums, existIndexs=[])
        return res

    # 45.子集:https://leetcode-cn.com/problems/subsets/
    # todo 其实就是全部子序列  记得pop
    def subsets(self, nums):
        def dfs(res, track, candidates, begin):
            if begin == len(candidates):
                res.append(track[:])
                return
            # 当前不选
            dfs(res, track, candidates, begin + 1)
            # 当前选择
            track.append(candidates[begin])
            dfs(res, track, candidates, begin + 1)
            track.pop()
            return
        res = []
        dfs(res, track=[], candidates=nums, begin=0)
        return res

    # 93.岛屿数量:https://leetcode-cn.com/problems/number-of-islands/
    # todo DFS或BFS
    def numIslands(self, grid: List[List[str]]) -> int:
        # dfs就是在把遍历到的岛清零避免重复计算
        def dfs(grid, i, j):
            if i < 0 or i > len(grid) - 1 or j < 0 or j > len(grid[0]) -1:
                return
            if grid[i][j] != '1':
                return
            else:
                grid[i][j] = '2'
                dfs(grid, i - 1, j)
                dfs(grid, i + 1, j)
                dfs(grid, i, j - 1)
                dfs(grid, i, j + 1)
                return
        num = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '1':
                    dfs(grid, i, j)
                    num += 1
        return num

    # 46.单词搜索:https://leetcode-cn.com/problems/word-search/
    # todo 记忆！图上搜索！visited的设置 + dfs的定义 ijk
    # todo 没做出来，跳了！
    def exist(self, board: List[List[str]], word: str) -> bool:
        pass

    # 68.分割回文串:https://leetcode-cn.com/problems/palindrome-partitioning/
    # todo 不会做
    def partition(self, s: str) -> List[List[str]]:
        pass
    # ---------------------------------------------------------------------------- #

    # -------------------------------- 二分 ----------------------------------- #
    # 33. 搜索旋转排序数组
    # todo 模型  c...... d a.....b     a<b<c<d   a最小，d最大
    #  mid两种情况： mid      mid
    #  1.mid落在[c,d]上,就nums[0]<=num[mid]：即[0,mid]有序，
    #  2.mid 在[a,b]，即[0,mid]无序
    def search(self, nums: List[int], target: int) -> int:
        pass

    # 34. 在排序数组中查找元素的第一个和最后一个位置
    # todo 就是找两个边界
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        pass

    # 69. x 的平方根
    # todo i*i<x  (i+1)*(i+1)>x  得i
    # todo 就是左边界
    def mySqrt(self, x: int) -> int:
        left, right = 0, x
        mid = left + ((right - left) >> 1)
        while(left <= right):
            mid = left + ((right - left) >> 1)
            if mid * mid > x:
                right = mid - 1
            elif mid * mid < x:
                left = mid + 1
            elif mid * mid == x:
                return mid
            else:
                pass
        # left = right + 1
        # todo r-1->i->r->l   或   r->l->i->l+1
        # todo mid-1<i<mid     或      mid<i<mid+1
        # todo (mid-1)^2<x<mid^2 或  (mid)^2<x<(mid+1)^2
        return mid if (mid * mid) < x else (mid - 1)


    # ---------------------------------------------------------------------------- #



if __name__ == '__main__':
    solution = Solution()
    a = [4,3,5,6,2]
    print(a.index(3))