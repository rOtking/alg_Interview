
# 冒泡ok
def bubbleSort(nums):
    isSwap = True
    for end in range(len(nums) - 1, -1, -1):
        if not isSwap:
            break
        isSwap = False
        for i in range(end - 1):
            if nums[i] > nums[i + 1]:
                nums[i], nums[i + 1] = nums[i + 1], nums[i]
                isSwap = True

# 选择ok
def selectSort(nums):
    for end in range(len(nums) - 1, -1, -1):
        max_index = end
        for i in range(end):
            if nums[i] > nums[max_index]:
                max_index = i
        nums[end], nums[max_index] = nums[max_index], nums[end]

# 插入ok
def insertSort(nums):
    for i in range(len(nums)):
        for j in range(i - 1, -1, -1):
            if nums[j] > nums[j + 1]:
                nums[j], nums[j + 1] = nums[j + 1], nums[j]
            else:
                break

# 归并ok
def mergeSort(nums):
    def process(nums, left, right):
        if left == right:
            return
        mid = left + ((right - left) >> 1)
        process(nums, left, mid)
        process(nums, mid + 1, right)
        merge(nums, left, mid, right)
        return
    def merge(nums, left, mid, right):
        helps = []
        p1, p2 = left, mid + 1
        while(p1 <= mid and p2 <= right):
            if nums[p1] <= nums[p2]:
                helps.append(nums[p1])
                p1 += 1
            else:
                helps.append(nums[p2])
                p2 += 1
        while(p1 <= mid):
            helps.append(nums[p1])
            p1 += 1
        while(p2 <= right):
            helps.append(nums[p2])
            p2 += 1
        for i in range(len(helps)):
            nums[left + i] = helps[i]
        return

    process(nums, 0, len(nums) - 1)

# 快排ok
import random
def quickSort(nums):
    def process(nums,left, right):
        if left >= right:
            return
        random_index = left + int(random.random() * (right - left) + 1)
        nums[random_index], nums[right] = nums[right], nums[random_index]
        small_right, big_left = partition(nums, left, right)
        process(nums, left, small_right)
        process(nums, big_left, right)
        return
    def partition(nums, left, right):
        pivot = nums[right]
        small_right, big_left = left - 1, right
        i = left
        while(i < big_left):
            if nums[i] < pivot:
                small_right += 1
                nums[small_right], nums[i] = nums[i], nums[small_right]
                i += 1
            elif nums[i] == pivot:
                i += 1
            elif nums[i] > pivot:
                big_left -= 1
                nums[big_left], nums[i] = nums[i], nums[big_left]
            else:
                pass
        nums[i], nums[right] = nums[right], nums[i]
        big_left += 1
        return small_right, big_left


    process(nums, 0, len(nums) - 1)


# 堆排ok
def heapSort(nums):
    def heapify(nums, index, heapSize):
        left = 2 * index + 1
        while(left < heapSize):
            right = left + 1
            largest = right if right < heapSize and nums[right] > nums[left] else left
            largest = largest if nums[largest] > nums[index] else index
            if largest == index:
                break
            else:
                nums[largest], nums[index] = nums[index], nums[largest]
                index = largest
                left = 2 * index + 1
        return
    heapSize = len(nums)
    for i in range(len(nums) - 1,-1, -1):
        heapify(nums, i, heapSize)
    heapSize = len(nums) - 1
    while(heapSize > 0):
        nums[0], nums[heapSize] = nums[heapSize], nums[0]
        heapify(nums, 0, )
        heapSize -= 1
    return

# 基本二分ok
def binarySearch(arr, target):
    left, right = 0, len(arr) - 1
    while(left <= right):
        mid = left + ((right - left) >> 1)
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        elif arr[mid] > target:
            right = mid - 1
        else:
            pass
    return -1


# 二分左边界ok
def binarySearchLeftBoundry(arr, target):
    left, right = 0, len(arr) - 1
    while(left <= right):
        mid = left + ((right - left) >> 1)
        if arr[mid] >= target:
            right = mid - 1
        else:
            left = mid + 1

    if 0 <= left <= len(arr) - 1 and arr[left] == target:
        return left
    else:
        return -1

# 二分右边界ok
def binarySearchRightBoundry(arr, target):
    left, right = 0, len(arr) - 1
    while(left <= right):
        mid = left + ((right - left) >> 1)
        if arr[mid] <= target:
            left = mid + 1
        else:
            right = mid - 1
    if 0 <= right <= len(arr) - 1 and arr[right] == target:
        return right
    else:
        return -1

# topk--python api ok
import heapq
def topK_api(arr, k):
    heap = arr[:k]
    heapq.heapify(heap)
    for i in range(k, len(arr)):
        if arr[i] > heap[0]:
            heap[0] = arr[i]
            heapq.heapify(heap)
    print(heap)

# topK 自己实现堆ok
def topK_heap(arr, k):
    def heapify(arr, index, heapSize):
        left = 2 * index + 1
        while(left < heapSize):
            right = left + 1
            smallest = right if right < heapSize and arr[right] < arr[left] else left
            smallest = smallest if arr[smallest] < arr[index] else index
            if smallest == index:
                return
            arr[smallest], arr[index] = arr[index], arr[smallest]
            index = smallest
            left = 2 * index + 1
        return
    heap = arr[:k]
    for i in range(len(heap) - 1, -1, -1):
        heapify(heap, i, len(heap))
    for i in range(k, len(arr)):
        if arr[i] > heap[0]:
            heap[0] = arr[i]
            heapify(heap, 0, len(heap))
    print(heap)

# topK partition ok
def topK_partition(arr, k):
    def partition(arr, left, right):
        if left >= right:
            return
        small_right, big_left = left - 1, right
        i = left
        pivot = arr[right]
        while(i < big_left):
            if arr[i] < pivot:
                small_right += 1
                arr[small_right], arr[i] = arr[i], arr[small_right]
                i += 1
            elif arr[i] == pivot:
                i += 1
            elif arr[i] > pivot:
                big_left -= 1
                arr[big_left], arr[i] = arr[i], arr[big_left]
            else:
                pass
        arr[i], arr[right] = arr[right], arr[i]
        big_left += 1
        return small_right, big_left
    left, right = 0, len(arr) - 1
    while(left < right):
        small_right, big_left = partition(arr, left,right)
        if small_right + 1 <= len(arr) - k <= big_left - 1:
            print(arr[len(arr) - k:])
            return arr[len(arr) - k :]
        elif len(arr) - k <= small_right:
            right = small_right
        elif len(arr) - k >= big_left:
            left = big_left
        else:
            pass
    return


# 两数之和ok
def twoSum(nums, target):
    num2index = {}
    for i in range(len(nums)):
        need = target - nums[i]
        if need in num2index:
            return [num2index[need], i]
        num2index[nums[i]] = i

# 整数反转ok
def reverse(x):
    INT_MIN, INT_MAX = -2 ** 31, 2 ** 31 - 1
    pos= True if x >= 0 else False
    x = x if pos else  -x
    arr = list(str(x))
    arr = arr[::-1]
    s = ''.join(arr)
    num = int(s) if pos else -int(s)
    return num if INT_MIN<=num<=INT_MAX else 0


# 盛最多水的容器 n>=2 ok
def maxArea(height) -> int:
    left, right = 0, len(height) - 1
    maxWater = 0
    while(left < right):
        water = (right - left) * min(height[left], height[right])
        maxWater = max(maxWater,water)
        if height[left] > height[right]:
            right -= 1
        else:
            left += 1
    return maxWater

# 最长公共前缀 str: List[str]  str可能'' ok
def longestCommonPrefix(strs):
    if '' in strs:
        return ''
    # 先求最短长度
    min_lenght = 0
    for i in range(len(strs)):
        min_lenght = min(min_lenght, len(strs[i]))

    idx = 0
    while(idx < min_lenght):
        flag = True
        pivot = strs[0][idx]
        for i in range(1, len(strs)):
            flag = flag and (strs[i][idx] == pivot)
        if flag:
            idx += 1
        else:
            return strs[0][:idx]
    return strs[0][:idx]

# 合并区间ok
def merge1(intervals: List[List[int]]) -> List[List[int]]:
    intervals = sorted(intervals, key=lambda x:x[0])
    merge = [intervals[0]]
    for i in range(1, len(intervals)):
        if intervals[i][0] <= merge[-1][1]:
            merge[-1][1] = max(merge[-1][1], intervals[i][1])
        else:
            merge.append(intervals[i])
    return merge

# 合并两个有序数组 num1初始m+n ok
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
        elif nums1[p1] >= nums2[p2]:
            nums1[p] = nums1[p1]
            p1 -= 1
        else:
            nums1[p] = nums2[p2]
            p2 -= 1
        p -= 1
    return

# 轮转数组ok
def rotate(nums, k: int) -> None:
    def change(nums, left, right):
        if left == right:
            return
        while(left < right):
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1
        return
    k = k % len(nums)
    change(nums, 0, len(nums) - 1)
    change(nums, 0, k - 1)
    change(nums, k, len(nums) - 1)
    return

# 移动零ok
def moveZeroes(nums) -> None:
    fast, slow = 0, 0
    while(fast < len(nums)):
        if nums[fast] != 0:
            nums[fast], nums[slow] = nums[slow], nums[fast]
            slow += 1
        fast += 1
    return


# 实现stack ok
class MinStack:
    def __init__(self):
        self.stack = []
        self.minStack = [float('-inf')]
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

# 最大数：一个数组能组成的最大数 ok
import functools
def largestNumber(nums) -> str:
    def cmp(s1, s2):
        if int(s1 + s2) < int(s2 + s1):
            return 1
        else:
            return -1
    s = [str(x) for x in nums]
    s = sorted(s,key=functools.cmp_to_key(cmp))
    if s[0] == '0':
        return '0'
    res = ''.join(s)
    return res

# 电话号码的字母组合ok
def letterCombinations1(digits: str):
    def dfs(res, track, candidates, the_index):
        if the_index == len(candidates) - 1:
            res.append(track)
            return
        for candidate in candidates[the_index]:
            track += candidate
            dfs(res, track, candidates, the_index+1)
            track = track[:-1]
        return

    res = []
    if len(digits) == 0 or digits is None:
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
    for digit in digits:
        candidates.append(digit2phone[digit])
    dfs(res, track='', candidates=candidates, the_index=0)
    return res

# 括号生成ok
def generateParenthesis(n: int):
    def dfs(res, track, n, leftNum, rightNum):
        if len(track) == 2 * n:
            res.append(track)
            return
        if leftNum < n:
            track += '('
            dfs(res, track, n, leftNum + 1, rightNum)
            track = track[:-1]
        if rightNum < n:
            track.append(')')
            dfs(res, track, n, leftNum, rightNum + 1)
            track = track[:-1]
        return
    res = []
    dfs(res, track='', n=n, leftNum=0, rightNum=0)
    return res

# 全排列 nums: List[int]   res -> List[List[int]] ok
def permute(nums):
    def dfs(res, track, candidates, existIndexs):
        if len(track) == len(candidates):
            res.append(track[:])
            return
        for idx, candidate in enumerate(candidates):
            if idx not in existIndexs:
                existIndexs.append(idx)
                track.append(nums[idx])
                dfs(res, track, candidates, existIndexs)
                track.pop()
                existIndexs.pop()
        return
    res = []
    dfs(res, track=[], candidates=nums, existIndexs=[])
    return res


# 子集 todo 超出条件！ok
def subsets(nums):
    def dfs(res, track, candidates, the_index):
        if the_index == len(candidates):
            res.append(track)
            return
        # 不选当前
        dfs(res, track, candidates, the_index+1)
        # 选
        track.append(candidates[the_index])
        dfs(res, track, candidates, the_index+1)
        track.pop()
        return

    res = []
    dfs(res, track=[], candidates=nums, the_index=0)
    return res


# 搜索旋转排序数组 todo 一边有序后的条件！
def search(nums, target: int) -> int:
    left, right = 0, len(nums) - 1
    while(left <= right):
        mid = left + ((right - left) >> 1)
        if nums[mid] == target:
            return mid
        # 左边有序
        elif nums[mid] > nums[left]:
            if nums[left] <= target <= nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[mid + 1] <= target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1

    return -1

# 删除链表的倒数第 N 个结点 ok
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
def removeNthFromEnd(head, n: int) -> ListNode:
    dummy = ListNode(0)
    dummy.next = head
    slow, fast = dummy, dummy
    for _ in range(n):
        fast = fast.next
    while(fast.next):
        slow = slow.next
        fast = fast.next
    toDel = slow.next
    slow.next = toDel.next
    toDel.next = None
    return dummy.next

# 合并两个有序链表ok
def mergeTwoLists(l1: ListNode, l2: ListNode) -> ListNode:
    p1, p2 = l1, l2
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


# 合并K个升序链表ok
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
    return process(lists, 0, len(lists) - 1)

# 复制带随机指针的链表ok
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random

def copyRandomList(head: 'Node') -> 'Node':
    if head is None:
        return None
    old2new = {}
    cur = head
    dummy = ListNode(0)
    pre = dummy
    while(cur):
        pre.next = Node(cur.val)
        pre = pre.next
        old2new[cur] = pre
        cur = cur.next
    p1, p2 = head, dummy.next
    while(p1):
        p2_random = old2new[p1.random]
        p2.random = p2_random
        p1 = p1.next
        p2 = p2.next
    return dummy.next


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
# 二叉树的中序遍历ok
def inorderTraversal1(root: TreeNode):
    res = []
    if root is None:
        return res
    res1 = inorderTraversal1(root.left)
    res2 = inorderTraversal1(root.right)
    res.extend(res1)
    res.append(root.val)
    res.extend(res2)
    return res
# 迭代ok
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

# 前序 迭代ok
def preorderTraversal(root: TreeNode):
    res = []
    stack = [root] if root else []
    while(len(stack) != 0):
        cur = stack.pop()
        res.append(cur.val)
        if cur.right:
            stack.append(cur.right)
        if cur.left:
            stack.append(cur.left)
    return res

# 后序 迭代
def postorderTraversal(root: TreeNode):
    res = []
    stack = [root] if root else []
    while(len(stack) != 0):
        cur = stack.pop()
        res.append(cur.val)
        if cur.left:
            stack.append(cur.left)
        if cur.right:
            stack.append(cur.left)
    return res[::-1]

# 验证二叉搜索树ok
def isValidBST(root: TreeNode) -> bool:
    def process(root):
        if root is None:
            return None
        res1 = process(root.left)
        res2 = process(root.right)
        leftOk = True if res1 is None or (res1[0] and res1[2] < root.val) else False
        rightOk = True if res2 is None or (res2[0] and res2[1] > root.val) else False
        return leftOk and rightOk, res1[1], res2[2]
    if root is None:
        return False
    return process(root)[0]

# 对称二叉树ok
def isSymmetric(root: TreeNode) -> bool:
    def check(p, q):
        if p is None and q is None:
            return True
        if p is None or q is None:
            return False
        res1 = check(p.left, q.right)
        res2 = check(p.right, q.left)
        return res1 and res2 and p.val == q.val

    if root is None:
        return True
    return check(root.left, root.right)

# 二叉树的层序遍历  -> List[List[int]]ok
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

# 二叉树的锯齿形层序遍历ok
def zigzagLevelOrder(root: TreeNode):
    res = []
    queue = [root] if root else []
    left2right = True  # 当前轮输出顺序
    while(len(queue)):
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
def maxDepth(root: TreeNode) -> int:
    if root is None:
        return 0
    leftDepth = maxDepth(root.left)
    rightDepth = maxDepth(root.right)
    return max(leftDepth, rightDepth) + 1

# 无重复字符的最长子串 todo left++ 的循环条件很多的
import collections
def lengthOfLongestSubstring1(s: str) -> int:
    left, right = 0, 0
    window = collections.defaultdict(int)
    maxSub = ''
    maxLenght = 0
    while(right < len(s)):
        if window[s[right]] == 0:
            window[s[right]] += 1
            right += 1
        if right - left > maxLenght:
            maxSub = s[left: right]
            maxLenght = right - left
        while(right < len(s) and left < right and window[s[right]] > 0):
            window[s[left]] -= 1
            left += 1
    return maxLenght

# 最长回文子串 todo ok更好了
def longestPalindrome(s: str) -> str:
    dp = [[True] * len(s) for _ in range(len(s))]
    for i in range(len(s)):
        dp[i][i] = True
    # dp[i][j]= if s[i] != s[j]:max(dp[i+1][j]下方, dp[i][j-1]左边) else dp[i+1][j-1] + 1左下角
    # 从下到上
    maxSub = s[0]
    for i in range(len(s) - 1, -1, -1):
        for j in range(i + 1, len(s)):
            if s[i] == s[j]:
                dp[i][j] = dp[i + 1][j - 1]
            else:
                dp[i][j] = False
            if dp[i][j] and j - i > len(maxSub):
                maxSub = s[i:j+1]
    return maxSub

# 编辑距离ok
def minDistance(word1: str, word2: str) -> int:
    dp = [[0] * (len(word2) + 1) for _ in range(len(word1) + 1)]
    for i in range(len(word1) + 1):
        dp[i][0] = i
    for j in range(len(word2) + 1):
        dp[0][j] = j
    for i in range(len(word1)):
        for j in range(len(word2)):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    return dp[-1][-1]

# 有效的括号:是否有效 todo stack[-1]可能是')'不在pair！！！
def isValid(s: str) -> bool:
    pair = {'(':')','[':']', '{':'}'}
    stack = []
    for ch in s:
        if len(stack) == 0:
            stack.append(ch)
        else:
            if stack[-1] in pair and pair[stack[-1]] == ch:
                stack.pop()
            else:
                stack.append(ch)
    return True if len(stack) == 0 else False


# 最小覆盖子串 todo 还是边界不清楚！！
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

# 字母异位词分组ok
def groupAnagrams(strs):
    hash = collections.defaultdict(list)
    for s in strs:
        key = sorted(s)
        hash[key].append(s)
    return list(hash.values())

# 从前序与中序遍历序列构造二叉树ok
def buildTree(preorder, inorder) -> TreeNode:
    if len(preorder) == 0:
        return None
    root_val = preorder[0]
    root_index = inorder.index(root_val)
    root = TreeNode(root_val)
    root.left = buildTree(preorder[1:root_index+1], inorder[:root_index])
    root.right = buildTree(preorder[root_index+1:], inorder[root_index+1:])
    return root


# 将有序数组转换为二叉搜索树ok
def sortedArrayToBST(nums) -> TreeNode:
    if len(nums) == 0:
        return None
    left, right = 0, len(nums) - 1
    mid = left + ((right - left) >> 1)
    root = TreeNode(nums[mid])
    root.left = sortedArrayToBST(nums[left:mid])
    root.right = sortedArrayToBST(nums[mid + 1:right])
    return root

# 二叉树的最大路径和:可以从下往上ok
def maxPathSum(root: TreeNode) -> int:
    # 求单链最大
    def subMax(root):
        nonlocal res
        if root is None:
            return 0
        res1 = max(subMax(root.left), 0)
        res2 = max(subMax(root.right), 0)
        res = max(res, root.val + res1 + res2)
        return root.val + max(res1, res2)
    if root is None:
        return 0
    res = float('-inf')
    _= subMax(root)
    return res

# 打家劫舍：数组不能相邻 todo i-3与i-2 ok
def rob(nums) -> int:
    if len(nums) == 1 or len(nums) == 2:
        return max(nums)
    dp = [0] * len(nums)
    dp[0], dp[1] = nums[0], nums[1]
    for i in range(2, len(nums)):
        dp[i] = max(dp[i - 2], dp[i - 3]) + nums[i]
    return max(dp)

# 打家劫舍 III：二叉树 至少一个结点ok
def rob3(root: TreeNode) -> int:
    def dp1(root):
        if root is None:
            return 0
        return root.val + dp2(root.left) + dp2(root.right)
    def dp2(root):
        if root is None or (root.left is None and root.right is None):
            return 0
        return max(dp1(root.left), dp2(root.left)) + max(dp1(root.right), dp2(root.right))

    return max(dp1(root), dp2(root))

# 最长递增子序列ok
def lengthOfLIS1(nums) -> int:
    dp = [1] * len(nums)
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

# 不同的二叉搜索树 todo 递归
def numTrees(n: int) -> int:
    def dp(i, j):
        if i >= j:
            return 1
        res = 0
        for p in range(i, j + 1):
            res += (dp(i, p - 1) * dp(p + 1, j))
        return res
    return dp(1, n)

# 跳跃游戏：按每个位置值跳，能不能到最后ok
def canJump(nums) -> bool:
    farest = 0
    for i in range(len(nums)):
        if farest >= len(nums) - 1:
            return True
        if farest < i:
            return False
        else:
            farest = max(farest, i + nums[i])

# 最小路径和：(0,0)到(m,n)最小和ok
def minPathSum(grid) -> int:
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = grid[0][0]
    for i in range(1, m):
        dp[i][0] = dp[i - 1][0] + grid[i][0]
    for j in range(1, n):
        dp[0][j] = dp[0][j - 1] + grid[0][j]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]
    return dp[-1][-1]

# 二叉搜索树中第K小的元素ok
def kthSmallest(root, k: int) -> int:

    stack = []
    cur = root
    i = 0
    while(len(stack) != 0 or cur):
        if cur:
            stack.append(cur.val)
            cur = cur.left
        else:
            cur = stack.pop()
            i += 1
            if i == k:
                return cur.val
            cur = cur.right
    return

# 二叉树的最近公共祖先ok
def lowestCommonAncestor(root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    if root is None or root is p or root is q:
        return root
    res1 = lowestCommonAncestor(root.left, p, q)
    res2 = lowestCommonAncestor(root.right, p, q)

    if res1 and res2:
        return root
    else:
        return res1 if res1 else res2

# 翻转二叉树ok
def invertTree(root: TreeNode) -> TreeNode:
    if root is None:
        return root
    left = invertTree(root.left)
    right = invertTree(root.right)
    root.left = right
    root.right = left
    return root

# 二叉树的直径ok
def diameterOfBinaryTree(root: TreeNode) -> int:
    # 单链最多结点数
    def process(root):
        nonlocal res
        if root is None:
            return 0
        leftDepth = process(root.left)
        rightDepth = process(root.right)
        res = max(res, leftDepth + rightDepth + 1)
        return 1 + max(leftDepth, rightDepth)
    res = 0
    _ = process(root)
    return res - 1

# 路径总和 III:和为target的所有路径ok
def pathSum(root: TreeNode, sum: int) -> int:
    # 以root的路径
    def process(root, sum):
        if root is None:
            if sum == 0:
                return 1
            else:
                return 0
        leftNum = process(root.left, sum - root.val)
        rightNum = process(root.right, sum - root.val)
        return leftNum + rightNum
    if root is None:
        return 0
    return process(root, sum) + pathSum(root.left, sum) + pathSum(root.right, sum)

# 合并二叉树ok
def mergeTrees(root1: TreeNode, root2: TreeNode) -> TreeNode:
    if root1 is None:
        return root2
    if root2 is None:
        return root1
    left = mergeTrees(root1.left, root2.left)
    right = mergeTrees(root1.right, root2.right)
    root = TreeNode(root1.val + root2.val)
    root.left = left
    root.right = right
    return root

# 二叉树展开为链表 todo ok
def flatten(root: TreeNode) -> None:
    # 返回尾
    def process(root):
        if root is None:
            return None
        if root.left is None and root.right is None:
            return root

        left, right = root.left, root.right
        leftTail = process(left)
        rightTail = process(right)
        if leftTail is None:
            return rightTail if rightTail else root
        else:
            root.right = left
            leftTail.right = right
            return rightTail if rightTail else leftTail
    _ = process(root)
    return


class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
# 填充每个节点的下一个右侧节点指针ok
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

# 排序链表ok
def sortList1(head: ListNode) -> ListNode:
    def process(head):
        if head is None:
            return head
        slow, fast = head, head
        while(fast.next and fast.next.next):
            fast = fast.next.next
            slow = slow.next
        head1, head2 = head, slow.next
        slow.next = None
        head1 = process(head1)
        head2 = process(head2)
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

# 相交链表ok
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
    first = headA if longA > longB else headB
    second = headA if first is headB else headB
    diff = longB - longA if longB > longA else longA - longB
    for _ in range(diff):
        first = first.next
    while(first):
        if first is second:
            return first
        first = first.next
        second = second.next
    return first

# 爬楼梯:n个，一次1或2ok
def climbStairs(n: int) -> int:
    dp = [0] * (n + 1)
    dp[1] = 1
    dp[2] = 2
    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[-1]

# 买卖股票的最佳时机  买入1天卖出1天ok
def maxProfit(prices: List[int]) -> int:
    min_buy = prices[0]
    dp = [0] * len(prices)
    for i in range(1, len(prices)):
        dp[i] = max(prices[i] - min_buy, 0)
        min_buy = min(min_buy, prices[i])
    return max(dp)

# 买卖股票的最佳时机 IIok
def maxProfit2(prices: List[int]) -> int:
    max_profit = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            max_profit += (prices[i] - prices[i - 1])
    return max_profit


# 最大子数组和ok
def maxSubArray(nums: List[int]) -> int:
    dp = [0] * len(nums)
    dp = nums[0]
    for i in range(1, len(nums)):
        dp[i] = nums[i] + dp[i - 1] if dp[i - 1] > 0 else nums[i]
    return max(dp)

# 乘积最大子数组
def maxProduct(nums: List[int]) -> int:
    dp1 = [0] * len(nums)
    dp2 = [0] * len(nums)
    dp1[0] = nums[0]
    dp2[0] = nums[0]
    for i in range(1, len(nums)):
        dp1[i] = min(dp1[i - 1] * nums[i], dp2[i - 1] * nums[i])
        dp2[i] = max(dp1[i - 1] * nums[i], dp2[i - 1] * nums[i])
    return max(dp2)

# 不同路径：几种路径ok
def uniquePaths(m: int, n: int) -> int:
    dp = [[1] * n for _ in range(m)]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i -1][j] + dp[i][j - 1]
    return dp[-1][-1]

# 最长公共子序列：两str最长 todo 初始化与递推
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
    return dp[-1][-1]


# 反转链表ok
def reverseList(head: ListNode) -> ListNode:
    pre = None
    cur = head
    while(cur):
        theNext = cur.next
        cur.next = pre
        pre = cur
        cur = theNext
    return pre

# 回文链表ok
def isPalindrome(head) -> bool:
    if head is None:
        return False
    slow, fast = head, head
    while(fast.next and fast.next.next):
        fast = fast.next.next
        slow = slow.next
    tail = reverseList(slow.next)
    p1, p2 = head, tail
    while(p1 and p2 and p1 is not p2):
        if p1.val != p2.val:
            return False
        p1 = p1.next
        p2 = p2.next
    return True

# 两数相加ok
def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
    p1, p2 = l1, l2
    add_bit = 0
    dummy = ListNode(0)
    pre = dummy
    while(p1 and p2):
        val = p1.val + p2.val + add_bit
        add_bit = 0
        if val >= 10:
            val -= 10
            add_bit = 1
        pre.next = ListNode(val)
        pre = pre.next
        p1 = p1.next
        p2 = p2.next
    while(p1):
        val = p1.val + add_bit
        add_bit = 0
        if val >= 10:
            val -= 10
            add_bit = 1
        pre.next = ListNode(val)
        pre = pre.next
        p1 = p1.next
    while(p2):
        val = p2.val + add_bit
        if val >= 10:
            val -= 10
            add_bit = 1
        pre.next = ListNode(val)
        pre = pre.next
        p2 = p2.next
    if add_bit != 0:
        pre.next = ListNode(add_bit)
    return dummy.next

# 环形链表:判断是否有环ok
def hasCycle(head: ListNode) -> bool:
    if head is None or head.next is None:
        return False
    slow, fast = head, head
    while(fast.next and fast.next.next):
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            return True
    return False

# 环形链表 II:求入环结点ok
def detectCycle(head: ListNode) -> ListNode:
    if head is None or head.next is None:
        return False
    slow, fast = head, head
    while(fast.next and fast.next.next):
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            break
    if slow is not fast:
        return None
    p1, p2 = head, fast
    while(p1 is not p2):
        p1 = p1.next
        p2 = p2.next
    return p1

# x 的平方根:整数ok
def mySqrt1(x: int) -> int:
    if x == 0:
        return x
    left, right = 1, x
    while(left <= right):
        mid = left + ((right - left) >> 1)
        if mid * mid <= x < (mid + 1) ** 2:
            return mid
        elif mid ** 2 > x:
            right = mid - 1
        elif (mid + 1) ** 2 <= x:
            left = mid + 1
        else:
            pass
# 浮点数开方，保留3位小数ok
def sqrt(x):
    left, right = 0, x
    while(left <= right):
        mid = left + (right - left) / 2
        if -1e-1 < mid ** 2 - x <= 1e-3:
            print(mid)
            return mid
        elif mid ** 2 < x:
            left = mid
        elif mid ** 2 > x:
            right = mid
        else:
            pass

# 岛屿数量 todo
def numIslands(grid) -> int:
    def dfs(grid, i, j):
        if i < 0 or i > len(grid) - 1 or j < 0 or j > len(grid[0]):
            return
        if grid[i][j] != '1':
            return
        grid[i][j] = '2'
        dfs(grid, i, j - 1)
        dfs(grid, i, j + 1)
        dfs(grid, i - 1, j)
        dfs(grid, i + 1, j)
        return
    m, n = len(grid), len(grid[0])
    num = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] == '1':
                dfs(grid, i, j)
                num += 1

    return num

# 岛屿最大面积ok
def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
    def dfs(grid, i, j):
        if i < 0 or i > len(grid) - 1 or j < 0 or j > len(grid[0]):
            return
        if grid[i][j] != 1:
            return 0
        else:
            grid[i][j] = 2
            area = 1
            area += dfs(grid, i - 1, j)
            area += dfs(grid, i + 1, j)
            area += dfs(grid, i, j - 1)
            area += dfs(grid, i, j + 1)
            return area

    m, n = len(grid), len(grid[0])
    max_area = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                area = dfs(grid, i, j)
                max_area = max(max_area, area)
    return max_area

class BiListNode:
    def __init__(self, val):
        self.val = val
        self.pre = None
        self.next = None
# LRU 缓存   todo 优先级高的在end
class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.size = 0
        self.cache = {}
        self.dummy_head = BiListNode(0)
        self.dummy_tail = BiListNode(0)
        self.dummy_head.next = self.dummy_tail
        self.dummy_tail.pre = self.dummy_head

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        thePre = node.pre
        theNext = node.next
        thePre.next = theNext
        theNext.pre = thePre
        head = self.dummy_head.next
        self.dummy_head.next = node
        node.pre = self.dummy_head
        node.next = head
        head.pre = node
        return node.val
    def put(self, key: int, value: int) -> None:
        pass

    def _del_head(self):
        pass

    def _move_to_end(self, node):
    # 已存在的node移动到end
        pass
    def _add_to_end(self, node):
    # 添加一个新的弄的到end
        pass


if __name__ == '__main__':
    sqrt(16)










