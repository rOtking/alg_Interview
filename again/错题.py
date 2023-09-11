
# 子集ok
def subsets(nums):
    def dfs(res, track, candidates, the_index):
        if the_index == len(candidates):
            res.append(track[:])
            return
        dfs(res, track, candidates, the_index+1)
        track.append(candidates[the_index])
        dfs(res, track, candidates, the_index+1)
        track.pop()
        return

    res = []
    dfs(res, track=[], candidates=nums, the_index=0)
    return res

# 搜索旋转排序数组ok
def search(nums, target: int) -> int:
    left, right = 0, len(nums) - 1
    while(left <= right):
        mid = left + ((right - left) >> 1)
        if nums[mid] == target:
            return mid
        # 左边有序，右边先增后降
        if nums[mid] > nums[left]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1

import collections
# 无重复字符的最长子串ok
def lengthOfLongestSubstring1(s: str) -> int:
    window = collections.defaultdict(int)
    left, right = 0, 0
    max_sub = ''
    max_length = 0
    while(right < len(s)):
        if window[s[right]] <= 0:
            window[s[right]] += 1
            right += 1
        if right - left > max_length:
            max_length = right - left
            max_sub = s[left:right]
        while(right <  len(s) and window[s[right]] >= 1 and left < right):
            window[s[left]] -= 1
            left += 1
    return max_length


# 有效的括号:是否有效 todo )]}不是key
def isValid(s: str) -> bool:
    pair = {'(':')', '[':']', '{':'}'}
    stack = []
    for ch in s:
        if len(stack) != 0 and stack[-1] in pair and pair[stack[-1]] == ch:
            stack.pop()
        else:
            stack.append(ch)
    return True if len(stack) == 0 else False


# 最小覆盖子串  todo right是超出1个的, left不能等于right
def minWindow(s: str, t: str) -> str:
    def isFull(window, need):
        for k, v in need.items():
            if window[k] < v:
                return False
        return True
    need, window = collections.defaultdict(int), collections.defaultdict(int)
    for ch in t:
        need[ch] += 1
    left, right = 0, len(s) - 1
    min_sub = ''
    min_length = len(s) + 1
    while(right < len(s)):
        if not isFull(window, need):
            window[s[right]] += 1
            right += 1
        while(left < right and isFull(window, need)):
            if right - left < min_length:
                min_sub = s[left:right]
                min_length = min(min_length, right - left)
            window[s[left]] -= 1
            left += 1
    return min_sub


# 打家劫舍：数组不能相邻ok
def rob(nums) -> int:
    dp = [0] *  len(nums)
    dp[0], dp[1] = nums[0], nums[1]
    for i in range(2, len(nums)):
        dp[i] = max(dp[i - 2], dp[i - 3]) + nums[i]
    return max(dp)

# 不同的二叉搜索树ok
def numTrees(n: int) -> int:
    def dp(i, j):
        if i >= j:
            return 1
        num = 0
        for i in range(1, n + 1):
            num += (dp(1, i - 1) * dp(i + 1, n))
        return num


    return dp(1, n)

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
# 二叉树展开为链表ok
def flatten(root: TreeNode) -> None:
    def process(root):
        if root is None:
            return root
        leftRoot, rightRoot = root.left, root.right
        leftTail = process(leftRoot)
        rightTail = process(rightRoot)
        if leftRoot is None:
            return rightTail if rightTail else root
        else:
            root.right = leftRoot
            if rightRoot:
                leftTail.right = rightRoot
                return rightTail
            else:
                return leftTail
    if root is None:
        return
    _ = process(root)
    return


# 最长公共子序列：两str最长ok
def longestCommonSubsequence(text1: str, text2: str) -> int:
    if len(text1) == 0 or len(text2) == 0:
        return 0
    dp = [[0] * (len(text2)) for  _  in range(len(text1))]
    for i in range(len(text1)):
        dp[i][0] = 1 if text2[0] in text1[:i + 1] else 0
    for j in range(len(text2)):
        dp[0][j] = 1 if text1[0] in text2[:j + 1] else 0
    for i in range(1, len(text1)):
        for j in range(1, len(text2)):
            if text1[i] == text2[j]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


# 岛屿数量ok
def numIslands(grid) -> int:
    def dfs(grid, i, j):
        if i < 0 or i > len(grid) - 1 or j < 0 or j > len(grid[0]) - 1:
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
            if grid == '1':
                num += 1
                dfs(grid, i, j)
    return num


# LRU 缓存 todo 看
class BiListNode:
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.pre = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.size = 0   # 当前容量
        self.cache = {}   # key2node
        self.dummy_head = BiListNode(0)  # 待删除
        self.dummy_tail = BiListNode(0)  # 最高优先级

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._moveToTail(node)
        return node.val

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            node = self.cache[key]
            node.val = value
            self.cache[key] = node
            self._moveToTail(node)
        else:
            if self.size < self.capacity:
                node = BiListNode(key, value)
                self.cache[key] = node
                self._addToTail(node)
            else:
                self._removeTheHead()
                node = BiListNode(key, value)
                self.cache[key] = node
                self._addToTail(node)

    # 容量超了，删除1个
    def _removeTheHead(self):
        removeNode = self.dummy_head.next
        self.cache.pop(removeNode.key)
        tmp = removeNode.next
        self.dummy_head.next = tmp
        tmp.pre = self.dummy_head
        removeNode.next = None
        removeNode.pre = None
        del removeNode
    # 调到最高优先级
    def _moveToTail(self, node):
        thePre, theNext = node.pre, node.next
        thePre.next = theNext
        theNext.pre = thePre
        self._addToTail(node)

    # 添加一个新node到最高优先级
    def _addToTail(self, node):
        thePre = self.dummy_tail.pre
        thePre.next = node
        node.next = self.dummy_tail
        self.dummy_tail.pre = node
        node.pre = thePre

# Pow(x, n)快速幂 todo
def myPow(self, x: float, n: int) -> float:
    def dfs(base, n):
        if n == 0:
            return 1.0
        y = dfs(base, n // 2)
        if n % 2 == 0:
            return y * y
        else:
            return y * y * base
    return dfs(x, n) if n >= 0 else 1.0 / dfs(x, -n)