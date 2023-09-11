# 冒泡排序
# 一遍ok
def bubbleSort(arr):
    isSwap = True
    for end in range(len(arr)- 1, -1, -1):
        if not isSwap:
            break
        isSwap = False
        for i in range(end):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                isSwap = True

    print(arr)

# 选择排序
# 注意 max_index = end
def selectSort(arr):
    for end in range(len(arr) - 1, -1, -1):
        max_index = end
        for i in range(end):
            if arr[i] > arr[max_index]:
                max_index = i
        arr[end], arr[max_index] = arr[max_index], arr[end]
    print(arr)

# 插入排序
# 0-j是有序取，i之后是无序：判断arr[j]与arr[j+1]的关系
def insertSort(arr):
    for i in range(len(arr)):
        for j in range(i - 1, -1, -1):
            if arr[j] <= arr[j + 1]:
                break
            else:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    print(arr)

# 归并排序
# 一遍ok 注意稳定性
def mergeSort(arr):
    def process(arr, left, right):
        if left == right:
            return
        mid = left + ((right - left) >> 1)
        process(arr, left, mid)
        process(arr, mid + 1, right)
        merge(arr, left, mid, right)
        return
    def merge(arr, left, mid, right):
        help = []
        i, j = left, mid + 1
        while(i <= mid and j <= right):
            if arr[i] <= arr[j]:
                help.append(arr[i])
                i += 1
            else:
                help.append(arr[j])
                j += 1
        while(i <= mid):
            help.append(arr[i])
            i += 1
        while(j <= right):
            help.append(arr[j])
            j += 1
        for i in range(len(help)):
            arr[left + i] = help[i]
        return
    process(arr, 0, len(arr) - 1)
    print(arr)

# 快速排序
import random
# 终止条件 left >= right
def quickSort(arr):
    def process(arr, left, right):
        # 如果partition后没有大于区或没有小于区，那会越界1个
        if left >= right:
            return
        random_index = left + int(random.random() * (right - left + 1))
        arr[random_index], arr[right] = arr[right], arr[random_index]
        equal_left, equal_right = partition(arr, left, right)
        process(arr, left, equal_left - 1)
        process(arr, equal_right + 1, right)
        return

    def partition(arr, left, right):
        small_right, big_left = left - 1, right
        i = left
        pivot = arr[right]
        while(i < big_left):
            if arr[i] < pivot:
                small_right += 1
                arr[small_right], arr[i] = arr[i], arr[small_right]
                i += 1     # 此时arr[i]是0，所以+1
            elif arr[i] == pivot:
                i += 1
            elif arr[i] > pivot:
                big_left -= 1
                arr[big_left], arr[i] = arr[i], arr[big_left]
            else:
                pass
        arr[big_left], arr[right] = arr[right], arr[big_left]
        equal_left, equal_right = small_right + 1, big_left
        return equal_left, equal_right

    process(arr, 0, len(arr) - 1)
    print(arr)

# 堆排序
# heapSize：从0位置开始几个元素代表堆，怎么赋值，什么时候停
def heapify(arr, index, heapSize):
    left = 2 * index + 1
    while(left < heapSize):
        right = left + 1
        largest = right if right < heapSize and arr[right] > arr[left] else left
        largest = largest if arr[largest] > arr[index] else index
        if index == largest:
            break
        arr[largest], arr[index] = arr[index], arr[largest]
        index = largest
        left = 2 * index + 1
def heapSort(arr):
    for i in range(len(arr) - 1, -1, -1):
        heapify(arr, i, len(arr))
    heapSize = len(arr) - 1
    while(heapSize > 0):
        arr[0], arr[heapSize] = arr[heapSize], arr[0]
        heapify(arr, 0, heapSize)
        heapSize -= 1
    print(arr)

# 基本二分
def binarySearch(arr, target):
    left, right = 0, len(arr) - 1
    while(left <= right):
        mid = left + ((right - left) >> 1)
        if arr[mid] == target:
            print(mid)
            return mid
        elif arr[mid] < target:
            left = mid + 1
        elif arr[mid] > target:
            right = mid - 1
        else:
            pass
    print(-1)
    return -1

# 二分左边界
# 最终left = right + 1
def binarySearchLeftBoundry(arr, target):
    left, right = 0, len(arr) - 1
    while(left <= right):
        mid = left + ((right - left) >> 1)
        if arr[mid] == target:
            right = mid - 1
        elif arr[mid] < target:
            left = mid + 1
        elif arr[mid] > target:
            right = mid - 1
        else:
            pass
    if left >= 0 and left <= len(arr) - 1 and arr[left] == target:
        print(left)
        return left
    return -1

# topk--python api
import heapq
def topK_api(arr, k):
    help = arr[:k]
    heapq.heapify(help)
    for i in range(k, len(arr)):
        if arr[i] > help[0]:
            help[0] = arr[i]
            heapq.heapify(help)
    print(help)

# topK 自己实现堆
# 注意实现的是大根堆   注意条件：while(left < heapSize)
def topK_heap(arr, k):
    def heapify(arr, index, heapSize):
        left = 2 * index + 1
        while(left < heapSize):
            right = left + 1
            smallest = right if right < heapSize and arr[right] < arr[left] else left
            smallest = index if arr[index] < arr[smallest] else smallest
            if smallest == index:
                break
            arr[smallest], arr[index] = arr[index], arr[smallest]
            index = smallest
            left = 2 * index + 1

    heap = arr[:k]
    for i in range(len(heap) - 1, -1, -1):
        heapify(heap, i, len(heap))

    for i in range(k, len(arr)):
        if arr[i] > heap[0]:
            heap[0] = arr[i]
            heapify(heap, 0, len(heap))
    print(heap)

# topK partition
# 循环二分parition即可
def topK_partition(arr, k):
    def partition(arr, left, right):
        small_right, big_left = left - 1, right
        i = left
        pivot = arr[right]
        while(i < big_left):
            if arr[i] < pivot:
                small_right += 1
                arr[small_right], arr[i] = arr[i], arr[small_right]
                i += 1     # 此时arr[i]是0，所以+1
            elif arr[i] == pivot:
                i += 1
            elif arr[i] > pivot:
                big_left -= 1
                arr[big_left], arr[i] = arr[i], arr[big_left]
            else:
                pass
        arr[big_left], arr[right] = arr[right], arr[big_left]
        equal_left, equal_right = small_right + 1, big_left
        return equal_left, equal_right
    left, right = 0, len(arr) - 1
    while(left <= right):
        equal_left, equal_right = partition(arr, left, right)
        if equal_left <= len(arr) - k <= equal_right:
            print(arr[(len(arr) - k):])
            return
        elif equal_left > len(arr) - k:
            right = equal_left - 1
        elif equal_right < len(arr) - k:
            left = equal_right + 1
        else:
            pass
    return

# 两数之和
# 一遍ok
def twoSum(nums, target):
    value2index = {}
    for i in range(len(nums)):
        need = target - nums[i]
        if need in value2index:
            return [i, value2index[need]]
        value2index[nums[i]] = i

# 整数反转
# 一遍ok
def reverse(x):
    INT_MIN = -2**31
    INT_MAX = 2**31 - 1
    pos = True if x >=0 else False
    x = x if x >= 0 else -x
    x = list(str(x))
    x = x[::-1]
    x = ''.join(x)
    x = int(x) if pos else -int(x)
    return x if INT_MIN<=x<=INT_MAX else 0

# 盛最多水的容器 n>=2
# 一遍ok，不考虑中间高的部分
def maxArea(height) -> int:
    left, right = 0, len(height) - 1
    maxWater = 0
    while(left < right):
        water = (right - left) * min(height[left], height[right])
        maxWater = max(maxWater, water)
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    return maxWater

# 最长公共前缀 str: List[str]  str可能''
# 一遍ok：两次遍历，先求最短长度
def longestCommonPrefix(strs):
    if '' in strs:
        return ''
    # 先看最短长度
    min_length = len(strs[0])
    for s in strs:
        min_length = min(min_length, len(s))
    s_index = 0
    while(s_index < min_length):
        flag = True
        pivot = strs[0][s_index]
        for s in strs:
            flag = flag and (s[s_index] == pivot)
        if not flag:
            return strs[0][:s_index]
        s_index += 1
    return strs[0][:s_index]


# 旋转图像：所有对角线互换，完事左右互换
# 合并区间：先按开始时间排序，当前的start与merged最后一个的end比，小就合并，end为两个end的max；大就直接加入
def merge1(intervals: List[List[int]]) -> List[List[int]]:
    nums = sorted(intervals, key=lambda x:x[0])
    merged = [nums[0]]
    for i in range(1, len(nums)):
        if nums[i][0] > merged[-1][1]:
            merged.append(nums[i])
        else:
            merged[-1][1] = max(nums[i][1], merged[-1][1])
    return merged

# 合并两个有序数组 num1初始m+n
# ok但是注意一个走完的处理
def merge(nums1, m, nums2, n) -> None:
    p1, p2 = m - 1, n - 1
    p = len(nums1) - 1
    while(p >= 0):
        if p1 < 0:
            nums1[p] = nums2[p2]
            p2 -= 1
        elif p2 < 0:
            nums1[p] = nums1[p1]
            p1 -= 1
        elif nums1[p1] > nums2[p2]:
            nums1[p] = nums1[p1]
            p1 -= 1
        else:
            nums1[p] = nums2[p2]
            p2 -= 1
        p -= 1
    return


# 轮转数组
# ok，注意in-place的修改 nums[::-1]会deepcopy不行
def rotate(nums, k: int) -> None:
    def change(nums, left, right):
        while (left < right):
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1
    k = k % len(nums)
    change(nums, 0, len(nums) - 1)
    # nums = nums[::-1]
    change(nums, 0, k - 1)
    left, right = k, len(nums) - 1
    change(nums, k, len(nums) - 1)


# 移动零
# ok,继续记忆
def moveZeroes(nums) -> None:
    slow, fast = 0, 0
    while(fast < len(nums)):
        if nums[fast] != 0:
            nums[slow], nums[fast] = nums[fast], nums[slow]
            slow += 1
        fast += 1

# 实现stack，常数时间getMin
class MinStack:
    def __init__(self):
        self.stack = []
        self.minStack = [float('inf')]
    def push(self, val: int) -> None:
        self.stack.append(val)
        if val < self.minStack[-1]:
            self.minStack.append(val)
        else:
            tmp = self.minStack[-1]
            self.minStack.append(tmp)
    def pop(self) -> None:
        self.stack.pop()
        self.minStack.pop()

    def top(self) -> int:
        return self.stack[-1]
    def getMin(self) -> int:
        return self.minStack[-1]

# 最大数：一个数组能组成的最大数
# sorted重构，还是没想出来！
import functools
def largestNumber(nums) -> str:
    def cmp(str1, str2):
        res1 = int(str1 + str2)
        res2 = int(str2 + str1)
        if res1 < res2:
            return 1
        elif res1 > res2:
            return - 1
        else:
            return 0

    res = ''
    num_str = [str(x) for x in nums]
    num_str = sorted(num_str, key=functools.cmp_to_key(cmp))
    if num_str[0] == '0':
        return '0'
    for s in num_str:
        res += s
    return res

# 电话号码的字母组合
# 一次ok
def letterCombinations1(digits: str):
    # this_index是本轮可选择的位置
    def dfs(res, track, candidates, this_index):
        if len(track) == len(candidates):
            res.append(track)
            return

        for candidate in candidates[this_index]:
            track += candidate
            dfs(res, track, candidates, this_index + 1)
            track = track[:-1]
        return

    res = []
    if digits is None or len(digits) == 0:
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
    candidates = []
    for s in digits:
        candidates.append(digit2phone[s])
    dfs(res, track='', candidates=candidates, this_index=0)
    return res

# 括号生成
# 一次ok
def generateParenthesis(n: int):
    def dfs(res, track, n, leftNum, rightNum):
        if len(track) == 2 * n:
            res.append(track)
            return
        candidates = []
        if leftNum < n:
            candidates.append('(')
        if rightNum < leftNum:
            candidates.append(')')
        for candidate in candidates:
            track += candidate
            if candidate == '(':
                dfs(res, track, n, leftNum + 1, rightNum)
            else:
                dfs(res, track, n, leftNum, rightNum + 1)
            track = track[:-1]
        return

    res = []
    dfs(res, track='', n=n, leftNum=0, rightNum=0)
    return res

# 全排列 nums: List[int]   res -> List[List[int]]
# 一次ok
def permute(nums):
    def dfs(res, track, all_candidates, choosedIndexs):
        if len(track) == len(all_candidates):
            res.append(track[:])
            return
        for idx, candidate in enumerate(all_candidates):
            if idx not in choosedIndexs:
                track.append(candidate)
                choosedIndexs.append(idx)
                dfs(res, track, all_candidates, choosedIndexs)
                track.pop()
                choosedIndexs.pop()
        return
    res = []
    dfs(res, track=[], all_candidates=nums, choosedIndexs=[])
    return res

# 子集
# ok，但是注意终止条件是 超出了范围才停止！
def subsets(nums):
    # this_index 本轮选择要不要加入的位置
    def dfs(res, track, all_candidates, this_index):
        if this_index == len(all_candidates):
            res.append(track[:])
            return

        dfs(res, track, all_candidates, this_index + 1)
        track.append(all_candidates[this_index])
        dfs(res, track, all_candidates, this_index + 1)
        track.pop()
        return

    res = []
    dfs(res, track=[], all_candidates=nums, this_index=0)
    return res

# 岛屿数量(todo！！！)

# 搜索旋转排序数组 5 6 7 8 <9> 10 1 2 3
#                8 9 10 1 <2> 3 4 5 6
# 需要看，利用有序部分算确定值，缩小范围；无序的就是else
def search(nums, target: int) -> int:
    left, right = 0, len(nums) - 1
    while(left <= right):
        mid = left + ((right - left) >> 1)
        if nums[mid] == target:
            return mid
        # 左边有序
        if nums[mid] > nums[left]:
            if nums[left] <= target <= nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # 右边有序
        else:
            if nums[mid] <= target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
# 删除链表的倒数第 N 个结点
# 秒了。
def removeNthFromEnd(head, n: int) -> ListNode:
    dummy = ListNode(0)
    dummy.next = head
    slow, fast = dummy, dummy
    for _ in range(n):
        fast = fast.next
    while(fast.next):
        slow = slow.next
        fast = fast.next
    toRemove = slow.next
    theNext = toRemove.next
    slow.next = theNext
    toRemove.next = None
    return dummy.next

# 合并两个有序链表
# 秒了
def mergeTwoLists(l1: ListNode, l2: ListNode) -> ListNode:
    dummy = ListNode(0)
    pre = dummy
    p1, p2 = l1, l2
    while(p1 and p2):
        if p1.val <= p2.val:
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

# 合并K个升序链表
# 秒了！注意 lists == []
def mergeKLists(lists) -> ListNode:
    def process(lists, left, right):
        if left == right:
            return lists[left]
        mid = left + ((right - left) >> 1)
        head1 = process(lists, left, mid)
        head2 = process(lists, mid + 1, right)
        return merge(head1, head2)

    def merge(head1, head2):
        if head1 is None:
            return head2
        if head2 is None:
            return head1
        dummy = ListNode(0)
        pre = dummy
        p1, p2 = head1, head2
        while(p1 and p2):
            if p1.val <= p2.val:
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
        return dummy.nexts

    return process(lists, 0, len(lists) - 1)

class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random

# 复制带随机指针的链表
# map秒了！
def copyRandomList(head: 'Node') -> 'Node':
    if head is None:
        return None
    node2copy = {}
    cur = head
    dummy = Node(0)
    pre = dummy
    while(cur):
        newNode = Node(cur.val)
        pre.next = newNode
        pre = pre.next
        node2copy[cur] = newNode
        cur = cur.next
    cur1 = head
    cur2 = dummy.next
    while(cur1):
        randomNode1 = cur1.random
        randomNode2 = node2copy[randomNode1]
        cur2.random = randomNode2
        cur1 = cur1.next
        cur2 = cur2.next
    return dummy.next

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
# 二叉树的中序遍历
# 递归 秒了！
def inorderTraversal1(root: TreeNode):
    res = []
    if root is None:
        return res
    if root.left is None and root.right is None:
        return [root.val]
    res1 = inorderTraversal1(root.left)
    res2 = inorderTraversal1(root.right)
    res.extend(res1)
    res.append(root.val)
    res.extend(res2)
    return res

# 迭代 秒了！
def inorderTraversal(root: TreeNode):
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

# 前序 迭代 秒杀！
def preorderTraversal(root: TreeNode):
    res = []
    cur = root
    stack = [cur] if cur else []
    while(len(stack) != 0):
        cur = stack.pop()
        res.append(cur.val)
        if cur.right:
            stack.append(cur.right)
        if cur.left:
            stack.append(cur.left)
    return res

# 后序 迭代 秒杀！
def postorderTraversal1(root: TreeNode):
    tmp = []
    cur = root
    stack = [cur] if cur else []
    while(len(stack) != 0):
        cur = stack.pop()
        tmp.append(cur.val)
        if cur.left:
            stack.append(cur.left)
        if cur.right:
            stack.append(cur.right)
    res = tmp[::-1]
    return res

# 验证二叉搜索树
# 注意返回值
def isValidBST(root: TreeNode) -> bool:
    def process(root):
        if root is None:
            return None
        if root.left is None and root.right is None:
            return True, root.val, root.val
        res1 = process(root.left)
        res2 = process(root.right)
        isValidLeft = True if res1 is None or (res1[0] and root.val > res1[2]) else False
        isValidRight = True if res2 is None or (res2[0] and root.val < res2[1]) else False
        return isValidLeft and isValidRight, res1[1], res2[2]
    return process(root)[0]

# 对称二叉树
# 秒了！
def isSymmetric(root: TreeNode) -> bool:
    def check(p, q):
        if p is None and q is None:
            return True
        if p is None or q is None:
            return False
        res1 = check(p.left, q.right)
        res2 = check(p.right, q.left)
        return p.val == q.val and res1 and res2
    if root is None:
        return True
    return check(root.left, root.right)

# 二叉树的层序遍历  -> List[List[int]]
# 秒了！
def levelOrder(root: TreeNode):
    res = []
    queue = [root] if root else []
    while(len(queue) != 0):
        size = len(queue)
        tmp = []
        for _ in range(size):
            cur = queue.pop(0)
            tmp.append(cur.val)
            if cur.left:
                queue.append(cur.left)
            if cur.right:
                queue.append(cur.right)
        res.append(tmp)
    return res

# 二叉树的锯齿形层序遍历
# 秒杀！
def zigzagLevelOrder(root: TreeNode):
    res = []
    queue = [root] if root else []
    left2right = True
    while(len(queue) != 0):
        size = len(queue)
        tmp = []
        for _ in range(size):
            if left2right:
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

# 二叉树的最大深度
# 秒杀！
def maxDepth(root: TreeNode) -> int:
    if root is None:
        return 0
    leftDepth = maxDepth(root.left)
    rightDepth = maxDepth(root.right)

    return 1 + max(leftDepth, rightDepth)

# 无重复字符的最长子串
# 注意 第二个while的条件3个, while (right < len(s) and left < right and window[s[right]] > 0):
# right < len(s)为了right多1个，防止越界
# window[s[right]] > 0不是1，因为多出来的right并咩有加入window，只有window空才会加入
import collections
def lengthOfLongestSubstring1(s: str) -> int:
    window = collections.defaultdict(int)
    left, right = 0, 0
    max_sub = ''
    max_length = 0
    while (right < len(s)):
        if window[s[right]] < 1:
            window[s[right]] += 1
            right += 1
        if right - left > max_length:
            max_sub = s[left: right]
            max_length = right - left
        while (right < len(s) and left < right and window[s[right]] > 0):
            window[s[left]] -= 1
            left += 1
    return max_length


# 最长回文子串
# dp更好写！（1）初始化主副对角；（2）    # dp[i][j]取决于左下的dp[i+1][j-1]，从下向上，从右到左，注意终止条件
def longestPalindrome(s: str) -> str:
    # dp[i][j]为s[i..j]闭区间是否回文；i>j没有意义
    dp = [[False] * len(s) for _ in range(len(s))]
    max_sub = s[0]
    max_length = 1
    # 初始化主对角线与副对角(相邻两个相等就是回文)
    for i in range(len(s)):
        dp[i][i] = True
        if i < len(s) - 1:
            dp[i][i + 1] = True if s[i] == s[i + 1] else False
            if dp[i][i + 1]:
                max_sub = s[i:i + 2]
                max_length = 2
    # dp[i][j]取决于左下的dp[i+1][j-1]
    for i in range(len(s) - 3, -1, -1):
        for j in range(len(s) - 1, i + 1, -1):
            if s[i] == s[j]:
                dp[i][j] = dp[i + 1][j - 1]
            else:
                dp[i][j] = False
            if dp[i][j] and j - i + 1 > max_length:
                max_sub = s[i:j + 1]
                max_length = j - i + 1
    return max_sub

# 编辑距离
# 几乎秒！注意dp[0][0]是要代表一个是空''的情况 dp[i][j]是word[0,i)的也就是word[0,i-1]
def minDistance(word1: str, word2: str) -> int:
    dp = [[0] * (len(word2) + 1) for _ in range(len(word1) + 1)]
    for j in range(len(word2) + 1):
        dp[0][j] = j
    for i in range(len(word1) + 1):
        dp[i][0] = i

    for i in range(1, len(word1) + 1):
        for j in range(1, len(word2) + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[ - 1][j - 1])
    return dp[len(word1)][len(word2)]

# 有效的括号:是否有效
# 秒了！且更好了
def isValid(s: str) -> bool:
    pair = {'(':')','[':']', '{':'}'}
    stack = []
    for ch in s:
        if len(stack) != 0 and stack[-1] in pair and pair[stack[-1]] == ch:
            stack.pop()
        else:
            stack.append(ch)
    return True if len(stack) == 0 else False


# 最小覆盖子串
# 注意边界条件怎么设：
# if right - left <= min_length解决s='a',t='a'
# 初始 min_sub = ''
def minWindow(s: str, t: str) -> str:
    def isFull(window, need):
        for k, v in need.items():
            if window[k] < v:
                return False
        return True
    need, window = collections.defaultdict(int), collections.defaultdict(int)
    for ch in t:
        need[ch] += 1
    left, right = 0, 0
    min_sub = ''
    min_length = len(s)
    while(right < len(s)):
        if not isFull(window, need):
            window[s[right]] += 1
            right += 1
        while(isFull(window, need)):
            if right - left <= min_length:
                min_sub = s[left:right]
                min_length = min(min_length, right - left)
            window[s[left]] -= 1
            left += 1
    return min_sub

# 字母异位词分组
# 注意sorted(str)的结果是list，要''.join(list)转为str
def groupAnagrams(strs):
    ana2str = collections.defaultdict(list)
    for s in strs:
        key = sorted(s)
        ana2str[key].append(s)
    return list(ana2str.values())

# 从前序与中序遍历序列构造二叉树
# 秒了
def buildTree(preorder, inorder) -> TreeNode:
    if len(preorder) == 0:
        return None
    root = TreeNode(preorder[0])
    root_in_index = inorder.index(preorder[0])
    left = buildTree(preorder[1:(root_in_index + 1)], inorder[:root_in_index])
    right = buildTree(preorder[(root_in_index + 1):], inorder[(root_in_index + 1):])
    root.left = left
    root.right = right
    return root

# 将有序数组转换为二叉搜索树
# 秒了！
def sortedArrayToBST(nums) -> TreeNode:
    if len(nums) == 0:
        return None
    left, right = 0, len(nums) - 1
    mid = left + ((right - left) >> 1)
    val = nums[mid]
    root = TreeNode(val)
    leftTree = sortedArrayToBST(nums[:mid])
    rightTree = sortedArrayToBST(nums[mid+1 :])
    root.left = leftTree
    root.right = rightTree
    return root

# 二叉树的最大路径和
# 秒了！nonlocal 域
def maxPathSum(root: TreeNode) -> int:
    # 返回以root为根的单链最大值，更新root的全局最大
    def process(root):
        nonlocal res
        if root is None:
            return 0
        leftMax = max(process(root.left), 0)
        rightMax = max(process(root.right), 0)
        res = max(root.val + leftMax + rightMax, res)
        return root.val + max(leftMax, rightMax)

    res = float('-inf')
    process(root)
    return res

# 打家劫舍：数组不能相邻
# 秒了！
def rob(nums) -> int:
    if len(nums) == 1 or len(nums) == 2:
        return max(nums)
    dp = [0] * len(nums)
    # 初始化
    dp[0], dp[1] = nums[0], nums[1]
    # 2-3 = -1 初始化的dp[-1]是0，一举两得！
    for i in range(2, len(nums)):
        dp[i] = nums[i] + max(dp[i - 2], dp[i - 3])
    return max(dp)

# 打家劫舍 III：二叉树 至少一个结点
# 二叉树的结构不适合dp表，树形DP就是递归就好了！haha
# 算秒！ 先写暴力，再改memo，不然乱了
def rob3(root: TreeNode) -> int:
    # 偷当前root最大值
    def dp1(root):
        nonlocal memo1, memo2
        if root in memo1:
            return memo1[root]
        if root is None:
            return 0
        if root.left not in memo2:
            memo2[root.left] = dp2(root.left)
        if root.right not in memo2:
            memo2[root.right] = dp2(root.right)
        memo1[root] = root.val + memo2[root.left] + memo2[root.right]
        return memo1[root]
    # 不偷的最大
    def dp2(root):
        nonlocal memo1, memo2
        if root in memo2:
            return memo2[root]
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
        memo2[root] = max(memo1[root.left], memo2[root.left]) + max(memo1[root.right], memo2[root.right])
        return memo2[root]
    memo1, memo2 = {}, {}
    return max(dp1(root), dp2(root))


# 最长递增子序列
# 秒了！
def lengthOfLIS1(nums) -> int:
    dp = [1] * len(nums)
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[j] + 1, dp[i])

    return max(dp)

# 不同的二叉搜索树
# 其实和树的特点关系不大，dp[i][j] nums[i..j]能得到多少BST
# todo LR范围尝试，每个数字做root，左右递归，直接想dp不容易，从暴力改;
# LR范围上每个位置都能做root！
def numTrees(n: int) -> int:
    def dp(i, j):
        if i >= j:
            return 1
        res = 0
        for k in range(i, j + 1):
            leftNum = dp(i, k - 1)
            rightNum = dp(k + 1, j)
            res += (leftNum * rightNum)
        return res

    return dp(1, n)


# 跳跃游戏：按每个位置值跳，能不能到最后
# 秒了
def canJump(nums) -> bool:
    farest = 0
    for i in range(len(nums)):
        if farest >= len(nums) - 1:
            return True
        else:
            if farest >= i:
                farest = max(farest, i + nums[i])
            else:
                return False

# 零钱兑换:target与不重复coin[1,2,5]，最少几枚
# dp[i]为组成i要几枚, dp[i] = min(dp[i-1], dp[i-2], dp[i-5]) + 1

# 最小路径和：(0,0)到(m,n)最小和
# 秒了
def minPathSum(grid) -> int:
    dp = [[0] * len(grid[0]) for _ in range(len(grid))]
    dp[0][0] = grid[0][0]
    for j in range(1, len(grid[0])):
        dp[0][j] = dp[0][j - 1] + grid[0][j]
    for i in range(1, len(grid)):
        dp[i][0] = dp[i - 1][0] + grid[i][0]
    for i in range(1, len(grid)):
        for j in range(1, len(grid[0])):
            dp[i][j] = grid[i][j] + min(dp[i - 1][j], dp[i][j - 1])
    return dp[len(dp) - 1][len(dp[0]) - 1]

# 二叉搜索树中第K小的元素
# 秒了
def kthSmallest(root, k: int) -> int:
    num = 0
    stack = []
    cur = root
    while(len(stack) != 0 or cur):
        if cur:
            stack.append(cur)
            cur = cur.left
        else:
            cur = stack.pop()
            num += 1
            if num == k:
                return cur.val
            cur = cur.right

# 二叉树的最近公共祖先
# 秒了
def lowestCommonAncestor(root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    if root is None or root is p or root is q:
        return root

    res1 = lowestCommonAncestor(root.left, p, q)
    res2 = lowestCommonAncestor(root.right, p, q)

    if res1 and res2:
        return root
    else:
        return res1 if res1 else res2

# 翻转二叉树
# 秒了！
def invertTree(root: TreeNode) -> TreeNode:
    if root is None:
        return root
    left = invertTree(root.left)
    right = invertTree(root.right)
    root.left = right
    root.right = left
    return root


# 二叉树的直径
# 一边更新一边返回单链
def diameterOfBinaryTree(root: TreeNode) -> int:
    # 求最深单链node个数
    def process(root):
        nonlocal res
        if root is None:
            return 0
        leftNum = process(root.left)
        rightNum = process(root.right)
        res = max(res, leftNum + rightNum)
        return 1 + max(leftNum, rightNum)

    res = 0
    _ = process(root)
    return res - 1

# 二叉树的序列化与反序列化(todo)

# 路径总和 III:和为target的所有路径
# 秒了！嵌套！
def pathSum(root: TreeNode, sum: int) -> int:
    # 固定root
    def process(root, sum):
        if root is None:
            return 0
        need = sum - root.val
        left = process(root.left, need)
        right = process(root.right, need)
        return left + right + 1 if root.val == sum else left + right
    if root is None:
        return 0
    return process(root, sum) + pathSum(root.left, sum) + pathSum(root.right, sum)

# 合并二叉树
# 秒
def mergeTrees(root1: TreeNode, root2: TreeNode) -> TreeNode:
    if root1 is None:
        return root2
    if root2 is None:
        return root1
    root = TreeNode(root1.val + root2.val)
    left = mergeTrees(root1.left, root2.left)
    right = mergeTrees(root1.right, root2.right)
    root.left = left
    root.right = right
    return root

# 二叉树展开为链表
# process就是展平且返回tail，为None分开讨论清楚即可
def flatten(root: TreeNode) -> None:
    # 返回tail
    def process(root):
        if root is None:
            return None
        leftTail = process(root.left)
        rightTail = process(root.right)
        if leftTail:
            rightBak = root.right
            root.right = root.left
            leftTail.right = rightBak
            root.left = None

            return rightTail if rightTail else leftTail
        else:
            return rightTail if rightTail else root
    tail = process(root)

class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
# 填充每个节点的下一个右侧节点指针
# 秒了
def connect(root: 'Node') -> 'Node':
    queue = [root] if root else []
    while(len(queue) != 0):
        size = len(queue)
        tmp = []
        for _ in range(size):
            cur = queue.pop(0)
            tmp.append(cur)
            if cur.left:
                queue.append(cur.left)
            if cur.right:
                queue.append(cur.right)
        for i in range(len(tmp) - 1):
            tmp[i].next = tmp[i + 1]
    return root

# 排序链表
# 秒了
def sortList1(head: ListNode) -> ListNode:
    def process(head):
        if head is None or head.next is None:
            return head
        slow, fast = head, head
        while(fast.next and fast.next.next):
            slow = slow.next
            fast = fast.next.next
        newHead = slow.next
        slow.next = None
        head1 = process(head)
        head2 = process(newHead)
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
            if p1.val <= p2.val:
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

# 相交链表
# 秒了
def getIntersectionNode(headA: ListNode, headB: ListNode) -> ListNode:
    if headA is None or headB is None:
        return None
    p1, p2 = headA, headB
    longA, longB = 0, 0
    while(p1):
        longA += 1
        p1 = p1.next
    while(p2):
        longB += 1
        p2 = p2.next
    fast, slow = None, None
    if longA > longB:
        fast, slow = headA, headB
    else:
        fast, slow = headB, headA
    diff = longA - longB if longA > longB else longB - longA
    for _ in range(diff):
        fast = fast.next
    while(fast and fast is not slow):
        fast = fast.next
        slow = slow.next
    return fast

# 爬楼梯:n个，一次1或2
# 秒了
def climbStairs(n: int) -> int:
    dp = [0] * (n + 1)
    dp[1] = 1
    dp[2] = 2
    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[-1]

# 买卖股票的最佳时机  买入1天卖出1天
# 秒杀
def maxProfit(prices: List[int]) -> int:
    dp = [0] * len(prices)
    min_price = prices[0]
    for i in range(1, len(prices)):
        if prices[i] > min_price:
            dp[i] = prices[i] - min_price
        min_price = min(min_price, prices[i])
    return max(dp)

# 买卖股票的最佳时机 II
# 秒了
def maxProfit2(prices: List[int]) -> int:
    res = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            res += prices[i] - prices[i - 1]
    return res

# 最大子数组和
# 秒杀
def maxSubArray(nums: List[int]) -> int:
    dp = [0] * len(nums)
    dp[0] = nums[0]
    for i in range(1, len(nums)):
        if dp[i - 1] > 0:
            dp[i] = dp[i - 1] + nums[i]
        else:
            dp[i] = nums[i]
    return max(dp)

# 乘积最大子数组
# 秒杀
def maxProduct(nums: List[int]) -> int:
    dp1 = [0] * len(nums)
    dp2 = [0] * len(nums)
    dp1[0], dp2[0] = nums[0], nums[0]
    for i in range(1, len(nums)):
        dp1[i] = min(nums[i], nums[i] * dp1[i - 1], nums[i] * dp2[i - 1])
        dp2[i] = max(nums[i], nums[i] * dp1[i - 1], nums[i] * dp2[i - 1])
    return max(dp2)

# 不同路径：几种路径
# 秒
def uniquePaths(m: int, n: int) -> int:
    # dp[i][j] (0,0)到(i,j)
    dp = [[0] * n for _ in range(m)]
    for j in range(n):
        dp[0][j] = 1
    for i in range(m):
        dp[i][0] = 1
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    return dp[m - 1][n - 1]


# 最长公共子序列：两str最长
# 秒了。能在开始用基础判断解决的，不要在dp里增加维度，会变复杂。
def longestCommonSubsequence(text1: str, text2: str) -> int:
    if len(text1) == 0 or len(text2) == 0:
        return 0
    dp = [[0] * len(text2) for _ in range(len(text1))]
    for j in range(len(text2)):
        dp[0][j] = 1 if text1[0] in text2[:j+1] else 0
    for i in range(len(text1)):
        dp[i][0] = 1 if text2[0] in text1[:i+1] else 0
    for i in range(1, len(text1)):
        for j in range(1, len(text2)):
            if text1[i] == text2[j]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[len(text1) - 1][len(text2) - 1]

# 单词拆分：word能不能由dict中的生成
# todo dp[i]是s[0...i]可否被拆分   dp[i] = dp[j] is True and s[j...i] in wordDict

# 反转链表
# 秒了
def reverseList(head: ListNode) -> ListNode:
    pre = None
    cur = head
    while(cur):
        theNext = cur.next
        cur.next = pre
        pre = pre.next
        cur = theNext
    return pre

# 回文链表
# 不用看了

# 删除链表中的结点
# 秒了
def deleteNode(self, node):
    theNext = node.next
    node.val = theNext.val
    node.next = theNext.next
    theNext.next = None

# 两数相加
# 不写了

# 环形链表:判断是否有环
# 秒
def hasCycle(head: ListNode) -> bool:
    if head is None or head.next is None:
        return False
    slow, fast = head, head
    while(fast.next and fast.next.next):
        slow = slow.next
        fast = fast.next.next
        if fast is slow:
            return True
    return False

# 环形链表 II:求入环结点
# 秒
def detectCycle(head: ListNode) -> ListNode:
    if head is None or head.next is None:
        return None
    slow, fast = head, head
    while (fast.next and fast.next.next):
        slow = slow.next
        fast = fast.next.next
        if fast is slow:
            break
    if fast.next is None or fast.next.next is None:
        return None
    p1, p2 = head, slow
    while(p1 is not p2):
        p1 = p1.next
        p2 = p2.next
    return p1

# x 的平方根
# 直接        if mid * mid <= x and (mid + 1) ** 2 > x:
#  卡死范围就完了！  return mid
#
def mySqrt1(x: int) -> int:
    if x == 0:
        return 0
    left, right = 1, x
    while(left <= right):
        mid = left + int((right - left) / 2)
        if mid * mid <= x and (mid + 1) ** 2 > x:
            return mid
        elif mid * mid < x:
            left = mid + 1
        elif mid * mid > x:
            right = mid - 1
        else:
            pass

# 岛屿数量
def numIslands(grid) -> int:
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


# 岛屿最大面积
# 核心是遍历每个位置并改值
def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
    def dfs(grid, i, j):
        if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]):
            return 0
        if grid[i][j] != 1:
            return 0
        else:
            num = 1
            grid[i][j] = 2
            num += dfs(grid, i - 1, j)
            num += dfs(grid, i + 1, j)
            num += dfs(grid, i, j - 1)
            num += dfs(grid, i, j + 1)
            return num

    res = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:
                num = dfs(grid, i ,j)
                res = max(res, num)
    return res


#  LRU 缓存
# map + 双向List + dummy_head、dummy_tail






if __name__ == '__main__':
    # 排序
    # arr = [5,3,3,3,4,6,1,6,6,8,1,2]
    # heapSort(arr)

    # 二分
    # arr1 = [1,1,4,4,4,5,7,8]
    # binarySearchLeftBoundry(arr1,4)

    # topK
    # arr2 = [1,1,9,9,8,8,6,6,7,7]
    # topK_api(arr2, 6)
    # topK_heap(arr2, 6)
    # topK_partition(arr2, 6)
    minWindow(s="a", t="a")