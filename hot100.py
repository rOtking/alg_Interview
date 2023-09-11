import collections

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
        self.res1 = 1
    # -------------------------------- 链表10题 ----------------------------------- #
    # 2. 两数相加
    # todo add_bit的处理
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        add_bit = 0
        dummy = ListNode(0)
        pre = dummy
        p1, p2 = l1, l2
        while(p1 and p2):
            val = add_bit + p1.val + p2.val
            add_bit = 0
            if val >= 10:
                add_bit = 1
                val -= 10
            pre.next = ListNode(val)
            pre = pre.next
        while(p1):
            val = p1.val + add_bit
            add_bit = 0
            if val >= 10:
                add_bit = 1
                val -= 10
            pre.next = ListNode(val)
            pre = pre.next
        while(p2):
            val = p2.val + add_bit
            add_bit = 0
            if val >= 10:
                add_bit = 1
                val -= 10
            pre = pre.next
        if add_bit != 0:
            pre.next = ListNode(add_bit)
        return dummy.next

    # 21. 合并两个有序链表
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if l1 is None:
            return l2
        if l2 is None:
            return l1
        p1, p2 = l1, l2
        dummy = ListNode(0)
        pre = dummy
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
        if p2:
            pre.next = p2
        return dummy.next

    # 206. 反转链表
    # todo 前提：pre是以逆序的头，cur是待处理，pre与cur默认断开！
    def reverseList(self, head: ListNode) -> ListNode:
        if head is None:
            return head
        pre = None
        cur = head
        while(cur):
            theNext = cur.next
            cur.next = pre
            pre = cur
            cur = theNext
        return pre

    # 19. 删除链表的倒数第 N 个结点
    # todo 找倒数n+1 fast从dummy开始
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        dummy = ListNode(0)
        dummy.next = head
        slow, fast = dummy, dummy
        for _ in range(n):
            fast = fast.next
        while(fast.next):
            fast = fast.next
            slow = slow.next
        toRemove = slow.next
        slow.next = toRemove.next
        del toRemove
        return dummy.next

    # 23. 合并K个升序链表
    def mergeKLists(self, lists) -> ListNode:
        def process(lists, left, right):
            if left == right:
                return lists[left]
            mid = left + ((right - left) >> 1)
            res1 = process(lists, left, mid)
            res2 = process(lists, mid + 1, right)
            return merge(res1, res2)

        def merge(head1, head2):
            p1, p2 = head1, head2
            dummy = ListNode(0)
            pre = dummy
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
            if p2:
                pre.next = p2
            return dummy.next
        if len(lists) == 0:
            return None
        return process(lists, 0, len(lists) - 1)

    # 141. 环形链表
    def hasCycle(self, head: ListNode) -> bool:
        if head is None:
            return False
        slow, fast = head, head
        while(fast and fast.next):
            slow = slow.next
            fast = fast.next.next
            if slow is fast:
                return True
        return False

    # 148. 排序链表
    def sortList(self, head: ListNode) -> ListNode:
        def process(head):
            if head is None or head.next is None:
                return head
            slow, fast = head, head
            while(fast.next and fast.next.next):
                slow = slow.next
                fast = fast.next.next
            theNext = slow.next
            slow.next = None
            head1 = process(head)
            head2= process(theNext)
            return merge(head1, head2)

        def merge(head1, head2):
            if head1 is None:
                return head2
            if head2 is None:
                return head1
            p1, p2 = head1, head2
            dummy = ListNode(0)
            pre = dummy
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
            if p2:
                pre.next = p2
            return dummy.next
        return process(head)

    # 160. 相交链表
    # 辅助空间
    def getIntersectionNode1(self, headA: ListNode, headB: ListNode) -> ListNode:
        if headA is None or headB is None:
            return None
        stackA, stackB = [], []
        p1, p2 = headA, headB
        while(p1):
            stackA.append(p1)
            p1 = p1.next
        while(p2):
            stackB.append(p2)
            p2 = p2.next
        res = None
        while(min(len(stackA), len(stackB)) != 0):
            p1 = stackA.pop()
            p2 = stackB.pop()
            if p1 is p2:
                res = p1
            else:
                break
        return res

    # fast与slow
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        if headA is None:
            return headB
        if headB is None:
            return headA
        longA, longB = 0, 0
        p1, p2 = headA, headB
        while(p1):
            longA += 1
            p1 = p1.next
        while(p2):
            longB += 1
            p2 = p2.next
        diff = longA - longB if longA > longB else longB - longA
        longHead = headA if longA > longB else headB
        shortHead = headA if longHead is headB else headB
        fast, slow = longHead, shortHead
        for _ in range(diff):
            fast = fast.next
        while(fast and slow and fast is not slow):
            fast = fast.next
            slow = slow.next
        return slow

    # 142. 环形链表 II
    def detectCycle(self, head: ListNode) -> ListNode:
        if head is None or head.next is None:
            return None
        fast, slow = head, head
        while(fast and fast.next):
            slow = slow.next
            fast = fast.next.next
            if fast is slow:
                break
        if fast is None or fast.next is None:
            return None
        slow = head
        while(fast is not slow):
            fast = fast.next
            slow = slow.next
        return slow

    # 234. 回文链表
    def isPalindrome(self, head) -> bool:
        def reverseList(head):
            if head is None or head.next is None:
                return head
            pre = None
            cur = head
            while(cur):
                theNext = cur.next
                cur.next = pre
                pre = cur
                cur = theNext
            return pre
        slow, fast = head, head
        while(fast.next and fast.next.next):
            fast = fast.next.next
            slow = slow.next
        tail = reverseList(slow)
        p1, p2 = head, tail
        res = True
        while(p1 and p2):
            if p1.val != p2.val:
                res = False
                break
            p1 = p1.next
            p2 = p2.next
        _ = reverseList(tail)
        return res


    # ---------------------------------------------------------------------------- #

    # -------------------------------- 二叉树14题 ----------------------------------- #

    # 102. 二叉树的层序遍历
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        res = []
        if root is None:
            return res
        queue = [root]
        while(len(queue)):
            tmp = []
            size = len(queue)
            for _ in range(size):
                cur = queue.pop(0)
                tmp.append(cur.val)
                if cur.left:
                    queue.append(cur.left)
                if cur.right:
                    queue.append(cur.right)
            res.append(tmp)
        return res

    # 124. 二叉树中的最大路径和
    def maxPathSum(self, root: TreeNode) -> int:
        def subMax(root):
            if root is None:
                return 0

            leftMax = max(subMax(root.left), 0)
            rightMax = max(subMax(root.right), 0)

            self.res = root.val + leftMax + rightMax if root.val + leftMax + rightMax > self.res else self.res
            return root.val + max(leftMax, rightMax)

        subMax(root)
        return self.res

    # 226. 翻转二叉树
    def invertTree(self, root: TreeNode) -> TreeNode:
        if root is None or (root.left is None and root.right is None):
            return root
        left = self.invertTree(root.left)
        right = self.invertTree(root.right)
        root.left = right
        root.right = left
        return root

    # 543. 二叉树的直径
    # todo
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        # 返回深度：结点个数   DFS
        def process(root):
            if root is None:
                return 0
            leftMax = process(root.left)
            rightMax = process(root.right)
            self.res1 = 1 + leftMax + rightMax if 1 + leftMax + rightMax > self.res1 else self.res1
            return 1 + max(leftMax, rightMax)

        _ = process(root)
        return self.res1 - 1

    # 101. 对称二叉树
    # 转为求两个树pq是否对称
    def isSymmetric(self, root: TreeNode) -> bool:
        def check(p, q):
            if p is None and q is None:
                return True
            if p is None or q is None:
                return False
            res1 = check(p.left, q.right)
            res2 = check(p.right, q.left)
            return (p.val == q.val) and res1 and res2

        return check(root.left, root.right)

    # 104. 二叉树的最大深度
    def maxDepth(self, root: TreeNode) -> int:
        if root is None:
            return 0
        if root.left is None and root.right is None:
            return 1

        leftDepth = self.maxDepth(root.left)
        rightDepth = self.maxDepth(root.right)
        return 1 + max(leftDepth, rightDepth)

    # 105. 从前序与中序遍历序列构造二叉树
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        if len(preorder) == 0:
            return None
        # 可去掉
        # if len(preorder) == 1:
        #     return TreeNode(preorder[0])

        root_val = preorder[0]
        root_index = inorder.index(root_val)
        left = self.buildTree(preorder[1:root_index + 1], inorder[:root_index])
        right = self.buildTree(preorder[root_index + 1:], inorder[root_index + 1:])
        root = TreeNode(root_val)
        root.left = left
        root.right = right
        return root

    # 236. 二叉树的最近公共祖先
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if root is None or root is p or root is q:
            return root

        res1 = self.lowestCommonAncestor(root.left, p, q)
        res2 = self.lowestCommonAncestor(root.right, p, q)

        if res1 and res2:
            return root

        return res1 if res1 else res2

    # 297. 二叉树的序列化与反序列化
    # 空#，_来间隔
    # 反序列记忆
    # todo
    def serialize(self, root) -> str:
        if root is None:
            return '#_'
        res = str(root.val) + '_'
        res += self.serialize(root.left)
        res += self.serialize(root.right)
        return res
    def deserialize(self, data) -> TreeNode:
        def process(queue):
            cur = queue.pop(0)
            if cur is '#':
                return None
            root = TreeNode(int(cur))
            left = process(queue)
            right = process(queue)
            root.left = left
            root.right = right
            return root
        queue = []
        data1 = data.split('_')
        for x in data1:
            if x != '':
                queue.append(x)
        root = process(queue)
        return root

    # 98. 验证二叉搜索树
    # 对None的处理
    def isValidBST3(self, root: TreeNode) -> bool:
        def process(root):
            if root is None:
                return None
            if root.left is None and root.right is None:
                return True, root.val, root.val
            res1 = process(root.left)
            res2 = process(root.right)
            isValid1 = True if res1 is None or (res1[0] and root.val > res1[2]) else False
            isValid2 = True if res2 is None or (res2[0] and root.val < res2[1]) else False
            return isValid1 and isValid2, res1[1] if res1 else root.val, res2[2] if res2 else root.val
        return process(root)[0]

    # 437. 路径总和 III
    # 辅助：以某结点为root的所有路径
    # todo 重点看！有嵌套！
    def pathSum1(self, root: TreeNode, sum: int) -> int:
        def rootSum(root, target):
            if root is None:
                return 0
            res = 0
            if root.val == target:
                res += 1
            res += rootSum(root.left, target - root.val)
            res += rootSum(root.right, target - root.val)
            return res

        if root is None:
            return 0
        res = rootSum(root, sum)
        res += self.pathSum(root.left, sum)
        res += self.pathSum(root.right, sum)
        return res

    # 94. 二叉树的中序遍历
    # 迭代 消解左边界:顺序啥的！还是不熟！
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        stack = []
        cur = root
        while(len(stack) != 0 or cur):
            if cur:
                stack.append(cur)
                cur = cur.left
            else:
                cur = stack.pop()
                res.append(cur.val)
                cur = cur.right
        return res

    # 617. 合并二叉树
    # 可能None
    def mergeTrees(self, root1: TreeNode, root2: TreeNode) -> TreeNode:
        if root1 is None:
            return root2
        if root2 is None:
            return root1
        left = self.mergeTrees(root1.left, root2.left)
        right = self.mergeTrees(root1.right, root2.right)
        root = TreeNode(root1.val + root2.val)
        root.left = left
        root.right = right
        return root

    # 114. 二叉树展开为链表
    # 至少2个node
    # todo 要得到tail才能连接, return不为None很讲究，除了root有left无right的情况
    def flatten(self, root: TreeNode) -> None:
        # 返回tail
        def process(root):
            if root is None or (root.left is None and root.right is None):
                return root
            leftRoot = root.left
            rightRoot = root.right
            root.left = None
            root.right = None
            leftTail = process(leftRoot)
            rightTail = process(rightRoot)
            if leftRoot:
                leftTail.right = rightRoot
                root.right = leftRoot
            else:
                root.right = rightRoot
            # todo 尾结点不能是None，不然返回后不能连接
            return rightTail if rightTail else leftTail

        _ = process(root)
        return
    # 116. 填充每个节点的下一个右侧节点指针
    # todo BFS 完美二叉树连接next
    def connect(self, root: 'Node') -> 'Node':
        if root is None:
            return root
        queue = [root]
        while(len(queue) != 0):
            size = len(queue)
            pre = None
            for _ in range(size):
                cur = queue.pop(0)
                if pre is not None:
                    pre.next = cur
                pre = cur
                if cur.left:
                    queue.append(cur.left)
                if cur.right:
                    queue.append(cur.right)
        return root
    # ---------------------------------------------------------------------------- #


    # -------------------------------- 前缀树10题 ----------------------------------- #
    # 437. 路径总和 III
    def pathSum(self, root: TreeNode, sum: int) -> int:
        pass
    # ---------------------------------------------------------------------------- #

    # -------------------------------- DP10题 ----------------------------------- #
    # todo dp[i]以i位置为结束的最值；dp[i][j] i..j范围上的最值；dp[i][j] s1的i与s2的j上的最值
    # todo i从左到右的尝试，每个位置都是选或不选的结果
    # todo 又是dp的内容并不是直接解，间接解更容易建立递推关系！！

    # todo 技术总结：字符串子序列最值问题，都是dp；核心都是dp[]的含义，记几个核心套路：
    # todo 1. 最长递增子序列：dp[i]是以s[i]结尾的最长子序列长度；
    # todo 2. 2维dp[i][j]：
    #             （1）两个str或数组，如最长公共子序列：子数组s1[0...i]与s2[0...j]上，要求子序列的长度；
    #              (2)一个str或数组：如最长回文子序列：在s[i...j]上，要求的子序列的长度。
    # todo 若问题不是序列数组，是一个数，如完全平方数n的最少数量，则dp[i]就是所求，完全平方是i的最少！

    # 70. 爬楼梯
    # n >= 1
    # dp[i]为爬i的种数，dp[i] = dp[i - 1] + dp[i - 2]
    def climbStairs(self, n: int) -> int:
        dp = [0] * (n + 1)
        dp[1] = 1
        if n == 1:
            return dp[n]
        dp[2] = 2
        if n == 2:
            return dp[n]

        for i in range(3, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[n]



    # 121. 买卖股票的最佳时机
    # dp[i]为i时卖出的最大收益, 个数大于等于1
    def maxProfit(self, prices: List[int]) -> int:
        dp = [0] * len(prices)
        dp[0] = 0
        if len(prices) == 1:
            return 0
        min_price = prices[0]   # 记录之前的最小买入值
        for i in range(1, len(prices)):
            dp[i] = max(prices[i] - min_price, 0)
            min_price = min(prices[i], min_price)
        return max(dp)
    # 122. 买卖股票的最佳时机 II
    # 可多次买卖
    # todo dp挺麻烦的，记个贪心把：今天比昨天高就交易：今天-昨天 正负零，只加0；等价最有结果
    # 记忆吧
    def maxProfit1(self, prices: List[int]) -> int:
        total = 0
        for i in range(1, len(prices)):
            profit = prices[i] - prices[i - 1]
            if profit > 0:
                total += profit
        return total

    # 53. 最大子数组和
    # 大于等于1
    # todo dp[i] i为结尾的最大子数组和
    def maxSubArray(self, nums: List[int]) -> int:
        dp = [0] * len(nums)
        dp[0] = nums[0]

        for i in range(1, len(nums)):
            if dp[i - 1] < 0:
                dp[i] = nums[i]
            else:
                dp[i] = nums[i] + dp[i - 1]
        return max(dp)

    # 152. 乘积最大子数组
    # todo 涉及正负，两个表：暴力max与min得到结果！爽！
    def maxProduct(self, nums: List[int]) -> int:
        # dp[i]以i结尾的最大乘与最小乘，因为涉及负负为正
        dp_max = [0] * len(nums)
        dp_min = [0] * len(nums)
        dp_max[0] = nums[0]
        dp_min[0] = nums[0]

        for i in range(1, len(nums)):
            dp_max[i] = max(nums[i], nums[i] * dp_max[i - 1], nums[i] * dp_min[i - 1])
            dp_min[i] = min(nums[i], nums[i] * dp_max[i - 1], nums[i] * dp_min[i - 1])

        return max(dp_max)

    # 198. 打家劫舍
    # todo i可以不选i-2而选i-3
    def rob(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        if len(nums) == 2:
            return max(nums)
        # dp[i]是以i为结尾的最高金额
        dp = [0] * len(nums)
        dp[0] = nums[0]
        dp[1] = nums[1]
        for i in range(2, len(nums)):
            dp[i] = max(dp[i - 2], dp[i - 3]) + nums[i] if i - 3 >= 0 else dp[i - 2] + nums[i]
        return max(dp[len(dp) - 1], dp[len(dp) - 2])

    # 337. 打家劫舍 III
    # todo 两个过程 dp1[root]是选root可得最大值,dp2[root]为不选root可得, dp[root] = max(dp1[root], dp2[root])
    def rob1(self, root: TreeNode) -> int:
        def dp1(root):
            # 选择当前结点获取的最大值
            if root is None:
                return 0
            if root.left not in memo2:
                memo2[root.left] = dp2(root.left)
            if root.right not in memo2:
                memo2[root.right] = dp2(root.right)
            return memo2[root.left] + memo2[root.right] + root.val
        def dp2(root):
            # 不选当前结点的最大值
            if root is None:
                return 0
            if root.left not in memo1:
                memo1[root.left] = dp1(root.left)
            if root.left not in memo2:
                memo2[root.left] = dp2(root.left)
            if root.right not in memo1:
                memo1[root.right] = dp1(root.right)
            if root.right not in memo2:
                memo2[root.right] = dp2(root.right)
            return max(memo1[root.left], memo2[root.left]) + max(memo1[root.right], memo2[root.right])
        memo1 = {}
        memo2 = {}
        if root is None:
            return 0
        return max(dp1(root), dp2(root))

    # 300. 最长递增子序列
    # todo 求以第i个位置为结尾的子序列，最大长度是多少
    def lengthOfLIS(self, nums: List[int]) -> int:
        dp = [1] * len(nums)
        dp[0] = 1
        for i in range(1, len(nums)):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = dp[j] + 1 if dp[j] + 1 > dp[i] else dp[i]
        return max(dp)


    # 96. 不同的二叉搜索树
    # 暴力：每个数字做root，左右递归----超时
    # todo 递归 LR范围尝试
    def numTrees1(self, n: int) -> int:
        def dp(i, j):
            if i >= j:
                return 1

            res = 0
            for k in range(i, j + 1):
                leftNum = dp(i, k - 1)
                rightNum = dp(k + 1, j)
                res += (leftNum * rightNum)

            return res
        nums = [i for i in range(1, n+1)]
        return dp(0, len(nums) - 1)

    # todo DP优化 不写了
    def numTrees(self, n: int) -> int:
        nums = [i for i in range(1, n+1)]
        dp = [[0] * len(nums) for _ in range(len(nums))]
        pass

    # 55. 跳跃游戏
    # todo dp[i]为能否到达i，不写了
    # todo 贪心：遍历每个位置更新能到达的最有距离！精妙！farest
    def canJump1(self, nums: List[int]) -> bool:
        farest = 0
        for i in range(len(nums)):
            if farest >= len(nums) - 1:
                return True
            if i <= farest:
                farest = max(farest, i + nums[i])
            else:
                return False

    # 279. 完全平方数
    # 复杂不看
    def numSquares(self, n: int) -> int:
        pass

    # 322. 零钱兑换
    # 求最少硬币个数
    # todo i从左到右的尝试，每个位置都是选或不选的结果
    # todo dp[i]组成i要的最少数量 min(dp[i-金额1] + 1, dp[i - 金额2] + 1，..., dp[i - 金额n] + 1)
    def coinChange(self, coins: List[int], amount: int) -> int:
        pass

    # 64. 最小路径和
    # 超时
    def minPathSum1(self, grid: List[List[int]]) -> int:
        # (i,j)最小和到
        def dp(i, j):
            if i == 0 and j == 0:
                return grid[i][j]
            if i == 0 and j > 0:
                return grid[i][j] + dp(i, j - 1)
            if i > 0 and j == 0:
                return grid[i][j] + dp(i - 1, j)

            return grid[i][j] + min(dp(i, j - 1), dp(i - 1, j))

        m = len(grid)
        n = len(grid[0])
        return dp(m - 1, n - 1)
    # todo dp ok了
    def minPathSum(self, grid: List[List[int]]) -> int:
        dp = [[0] * len(grid[0]) for _ in range(len(grid))]
        dp[0][0] = grid[0][0]

        for j in range(1, len(grid[0])):
            dp[0][j] = dp[0][j - 1] + grid[0][j]
        for i in range(1, len(grid)):
            dp[i][0] = dp[i - 1][0] + grid[i][0]

        for i in range(1, len(grid)):
            for j in range(1, len(grid[0])):
                dp[i][j] = grid[i][j] + min(dp[i][j - 1], dp[i - 1][j])
        return dp[-1][-1]



    # 62. 不同路径
    # todo 简单
    def uniquePaths(self, m: int, n: int) -> int:
        # dp[i][j]从(0,0)到(i,j)不同数
        dp = [[0] * n for _ in range(m)]
        dp[0][0] = 1
        for j in range(1, n):
            dp[0][j] = 1
        for i in range(1, m):
            dp[i][0] = 1

        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]

        return dp[m - 1][n - 1]
    # 1143. 最长公共子序列
    # todo dp[i][j]是t1[...i]与t2[...j]的最长公共子序列
    # todo 需要记忆
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        if len(text1) == 0 or len(text2) == 0:
            return 0

        # base case
        dp = [[0] * len(text2) for _ in range(len(text1))]

        for i in range(len(text1)):
            if text2[0] in text1[:i+1]:
                dp[i][0] = 1

        for j in range(len(text2)):
            if text1[0] in text2[:j+1]:
                dp[0][j] = 1

        for i in range(1, len(text1)):
            for j in range(1, len(text2)):
                if text1[i] == text2[j]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[len(text1) - 1][len(text2) - 1]
    # 139. 单词拆分
    # todo dp[i]是s[0...i]可否被拆分   dp[i] = dp[j] is True and s[j...i] in wordDict
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        pass

    # 42. 接雨水
    # 太难 跳过
    def trap_dp(self, height: List[int]) -> int:
        pass

    # ---------------------------------------------------------------------------- #
    # -------------------------------- DFS回溯10题 ----------------------------------- #
    # 22. 括号生成
    def generateParenthesis(self, n: int):
        pass

    # ---------------------------------------------------------------------------- #

    # -------------------------------- 字符串6题 ----------------------------------- #
    # todo 貌似字符串window用的多，只有 编辑距离是DP，DP主要是数组的
    # todo str是子串问题，数组是子序列问题
    # 3. 无重复字符的最长子串
    # todo 记忆 滑窗
    def lengthOfLongestSubstring(self, s: str) -> int:
        if len(s) == 0:
            return 0
        max_length = 0
        window = set()
        left, right = 0, 0
        while(right < len(s)):
            if s[right] not in window:
                window.add(s[right])
                right += 1
            else:
                while(s[right] in window and left <= right):
                    window.remove(s[left])
                    left += 1
            max_length = max(len(window), max_length)
        return max_length

    # 或者window是map，存个数，left就可以判断window[s[right]] > 1就left++，进而去掉else了。
    def lengthOfLongestSubstring1(self, s: str) -> int:
        if len(s) == 0:
            return 0
        max_length = 0
        max_sub = ''
        window = collections.defaultdict(int)
        left, right = 0, 0
        while(right < len(s)):
            if window[s[right]] < 1:
                window[(s[right])] += 1
                right += 1
            if right - left > max_length:
                max_length = right - left
                max_sub = s[left: right]
            # todo 注意此时right是有可能超出去的，right是window之外下一个的！
            while(right < len(s) and window[s[right]] > 0 and left <= right):
                window[s[left]] -= 1
                left += 1
        return max_length
    # 5. 最长回文子串
    # todo 中心扩散 虚轴  abba  #a#b#b#a#
    # dp[i][j]为s[i...j]上是否回文,与dp[i+1][j-1]的关系
    # dp[i][j]为回文需要 s[i] == s[j] and dp[i+1][j-1] == True
    def longestPalindrome(self, s: str) -> str:
        def centerExpand(s, centor_index):
            left, right = centor_index, centor_index
            while(left >= 0 and right <= len(s) - 1):
                if s[left] != s[right]:
                    break
                left -= 1
                right += 1
            left += 1
            right -= 1
            return s[left:right + 1]
        chs = '#'
        for ch in s:
            chs += (ch + '#')
        max_sub = ''
        max_length = 0
        for centor_index in range(len(chs)):
            sub = centerExpand(chs, centor_index)
            if len(sub) > max_length:
                max_sub = sub
                max_length = len(max_sub)
        res = ''
        for ch in max_sub:
            if ch != '#':
                res += ch
        return res



    # 72. 编辑距离
    # todo 字符串的DP：核心是从结尾开始对A的 删换增操作，其实没有A增，A增等价为B删，知道A或B空，剩下的删掉即可！
    # todo 操作顺序不影响结果，所以从结尾开始好控制。
    # todo 超时的暴力算法是原型！最关键！
    def minDistance1(self, word1: str, word2: str) -> int:
        def dp(i, j):
            # i为word1的结尾位置，j是word2的
            # base case
            if i == -1:
                return j + 1
            if j == -1:
                return i + 1

            if word1[i] == word2[j]:
                return dp(i - 1, j - 1)
            else:
                # 删i
                delete = dp(i - 1, j) + 1
                # 换
                change = dp(i - 1, j - 1) + 1
                # 增 也就是删j
                add = dp(i, j - 1) + 1
                return min(delete, change, add)
        return dp(len(word1) - 1, len(word2) - 1)

    def minDistance2(self, word1: str, word2: str) -> int:
        def dp(i, j):
            if (i, j) in memo:
                return memo[(i, j)]
            # base case
            if i == -1:
                memo[(i, j)] = j + 1
                return memo[(i, j)]
            if j == -1:
                memo[(i, j)] = i + 1
                return memo[(i, j)]

            if word1[i] == word2[j]:
                memo[(i, j)] = dp(i - 1, j - 1)
                return memo[(i, j)]
            else:
                # 删i
                if (i - 1, j) not in memo:
                    memo[(i - 1, j)] = dp(i - 1, j)
                delete = memo[(i - 1, j)] + 1
                # 换
                if (i - 1, j - 1) not in memo:
                    memo[(i - 1, j - 1)] = dp(i - 1, j - 1)
                change = memo[(i - 1, j - 1)] + 1
                # 增 也就是删j
                if (i, j - 1) not in memo:
                    memo[(i, j - 1)] = dp(i, j - 1)
                add = memo[(i, j - 1)] + 1
                memo[(i, j)] = min(delete, change, add)
                return memo[(i, j)]
        memo = {}
        return dp(len(word1) - 1, len(word2) - 1)

    # todo 看着暴力递归写就行！注意有可能加一个行与下标！
    def minDistance3(self, word1: str, word2: str) -> int:
        dp = [[0] * (len(word2) + 1) for _ in range(len(word1) + 1)]
        # 1.所求为dp[5][3]
        # 初始化
        for j in range(len(word2) + 1):
            dp[0][j] = j
        for i in range(len(word1) + 1):
            dp[i][0] = i
        # 确定递推关系与顺序
        for i in range(1, len(word1) + 1):
            for j in range(1, len(word2) + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1
        return dp[len(word1)][len(word2)]


    # 20. 有效的括号
    # 至少1个
    # todo stack 消除与剩余来判断！
    def isValid(self, s: str) -> bool:
        def isPair(left, right):
            pair = {'(':')', '[':']', '{':'}'}
            if left in pair and pair[left] == right:
                return True
            else:
                return False

        stack = []
        for ch in s:
            if len(stack) == 0:
                stack.append(ch)
            else:
                if isPair(stack[-1], ch):
                    stack.pop()
                else:
                    stack.append(ch)
        if len(stack) == 0:
            return True
        else:
            return False

    # 76. 最小覆盖子串
    # todo window核心是：建立判断满足条件与不满足条件的 评判体系！这里是借助hash来判断
    # todo window的不满足就r++，只要满足就left++；isOk函数应该能优化，left++的while不要在else中
    def minWindow(self, s: str, t: str) -> str:
        def isOk(window, targetMap):
            # window是否满足
            for k, v in targetMap.items():
                if window[k] < v:
                    return False
            return True

        left, right = 0, 0
        targetMap = {} # 目标需要什么字符，分别是多少个
        window = {}    # 当前窗口内有多少
        for ch in t:
            window[ch] = 0
            if ch in targetMap:
                targetMap[ch] += 1
            else:
                targetMap[ch] = 1
        min_sub = ''
        min_length = float('inf')
        while(right < len(s)):
            if not isOk(window, targetMap):
                if s[right] in window:
                    window[s[right]] += 1
                right += 1
            # todo 不能加else，不然最后一轮left++进不来
            while(isOk(window, targetMap)):
                # 满足
                if min_length > right - left:
                    min_length = right - left
                    min_sub = s[left:right]
                if s[left] in window:
                    window[s[left]] -= 1
                left += 1
        return min_sub


    # 49. 字母异位词分组
    # todo 方法使用 核心是排序得到key与hash存
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        str_list = []
        mp = collections.defaultdict(list)
        for s in strs:
            key = ''.join(sorted(s))
            mp[key].append(s)

        return list(mp.values())

    # 10. 正则表达式匹配
    # todo DP
    def isMatch(self, s: str, p: str) -> bool:
        pass

    # 32. 最长有效括号
    # todo 太难不常考
    # ---------------------------------------------------------------------------- #
