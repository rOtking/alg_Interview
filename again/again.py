
# Pow(x, n)快速幂 todo
def myPow(x: float, n: int) -> float:
    pass


# 岛屿数量ok
def numIslands(grid) -> int:
    pass

# 最长公共子序列：两str最长ok
def longestCommonSubsequence(text1: str, text2: str) -> int:
    pass


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
# 二叉树展开为链表ok
def flatten(root: TreeNode) -> None:
    pass

# 不同的二叉搜索树ok
def numTrees(n: int) -> int:
    pass

import collections
# 最小覆盖子串
def minWindow(s: str, t: str) -> str:
    pass

# 打家劫舍：数组不能相邻ok
def rob(nums) -> int:
    pass


# 有效的括号
def isValid(s: str) -> bool:
    pass

# 无重复字符的最长子串ok
def lengthOfLongestSubstring1(s: str) -> int:
    pass


# 搜索旋转排序数组ok
def search(nums, target: int) -> int:
    pass


# 子集ok
def subsets(nums):
    pass

# 岛屿最大面积ok
def maxAreaOfIsland(self, grid) -> int:
    pass

# 浮点数开方，保留3位小数ok
def sqrt(x):
    pass


# x 的平方根:整数ok
def mySqrt1(x: int) -> int:
    pass


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

# 环形链表 II:求入环结点ok
def detectCycle(head: ListNode) -> ListNode:
    pass

# 环形链表:判断是否有环ok
def hasCycle(head: ListNode) -> bool:
    pass

# 两数相加ok
def addTwoNumbers(l1: ListNode, l2: ListNode) -> ListNode:
    pass


# 反转链表ok
def reverseList(head: ListNode) -> ListNode:
    pass


# 回文链表ok
def isPalindrome(head) -> bool:
    pass

# 不同路径：几种路径ok
def uniquePaths(m: int, n: int) -> int:
    pass

# 乘积最大子数组
def maxProduct(nums: List[int]) -> int:
    pass

# 最大子数组和ok
def maxSubArray(nums: List[int]) -> int:
    pass

# 买卖股票的最佳时机  买入1天卖出1天ok
def maxProfit(prices: List[int]) -> int:
    pass


# 买卖股票的最佳时机 IIok
def maxProfit2(prices: List[int]) -> int:
    pass


# 爬楼梯:n个，一次1或2ok
def climbStairs(n: int) -> int:
    pass


# 相交链表ok
def getIntersectionNode(headA: ListNode, headB: ListNode) -> ListNode:
    tailA = reverseList(headA)
    tailB = reverseList(headB)
    p1, p2 = tailA, tailB
    while(p1 and p2 and p1 is p2):
        p1 = p1.next
        p2 = p2.next
    headA = reverseList(tailA)
    headB = reverseList(tailB)
    return p1.next



# 排序链表ok
def sortList1(head: ListNode) -> ListNode:
    pass

class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
# 填充每个节点的下一个右侧节点指针ok
def connect(root: 'Node') -> 'Node':
    pass


# 合并二叉树ok
def mergeTrees(root1: TreeNode, root2: TreeNode) -> TreeNode:
    pass


# 路径总和 III:和为target的所有路径ok
def pathSum(root: TreeNode, sum: int) -> int:
    pass

# 二叉树的直径ok
def diameterOfBinaryTree(root: TreeNode) -> int:
    pass

# 翻转二叉树ok
def invertTree(root: TreeNode) -> TreeNode:
    pass


# 二叉树的最近公共祖先ok
def lowestCommonAncestor(root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    pass

# 二叉搜索树中第K小的元素ok
def kthSmallest(root, k: int) -> int:
    pass


# 最小路径和：(0,0)到(m,n)最小和ok
def minPathSum(grid) -> int:
    pass

# 跳跃游戏：按每个位置值跳，能不能到最后ok
def canJump(nums) -> bool:
    pass


# 最长递增子序列ok
def lengthOfLIS1(nums) -> int:
    pass

# 打家劫舍 III：二叉树 至少一个结点ok
def rob3(root: TreeNode) -> int:
    pass


# 二叉树的最大路径和:可以从下往上ok
def maxPathSum(root: TreeNode) -> int:
    pass

# 将有序数组转换为二叉搜索树ok
def sortedArrayToBST(nums) -> TreeNode:
    pass

# 从前序与中序遍历序列构造二叉树ok
def buildTree(preorder, inorder) -> TreeNode:
    pass

# 字母异位词分组ok
def groupAnagrams(strs):
    pass

# 编辑距离ok
def minDistance(word1: str, word2: str) -> int:
    pass

# 最长回文子串
def longestPalindrome(s: str) -> str:
    pass

# 二叉树的最大深度
def maxDepth(root: TreeNode) -> int:
    pass

# 二叉树的层序遍历  -> List[List[int]]ok
def levelOrder(root: TreeNode):
    pass


# 二叉树的锯齿形层序遍历ok
def zigzagLevelOrder(root: TreeNode):
    pass

# 对称二叉树ok
def isSymmetric(root: TreeNode) -> bool:
    pass


# 验证二叉搜索树ok
def isValidBST(root: TreeNode) -> bool:
    def process(root):
        if root is None:
            return True, None, None   # 最小，最大
        res1 = process(root.left)
        res2 = process(root.right)
        leftOk = True if res1[0] and (res1[2] is None or root.val > res2[2]) else False
        rightOk = True if res2[0] and (res2[1] is None or root.val < res2[1]) else False

        return leftOk and rightOk, res1[1] if res1[1] else root.val, res2[2] if res1[2] else root.val

    return process(root)[0]

# 二叉树的中序遍历ok
def inorderTraversal1(root: TreeNode):
    pass

# 迭代ok
def inorderTraversal(root: TreeNode):
    pass

# 前序 迭代ok
def preorderTraversal(root: TreeNode):
    pass

# 后序 迭代
def postorderTraversal(root: TreeNode):
    pass

# 复制带随机指针的链表ok
class Node1:
    def __init__(self, x: int, next: 'Node1' = None, random: 'Node1' = None):
        self.val = int(x)
        self.next = next
        self.random = random

def copyRandomList(head: 'Node1') -> 'Node1':
    pass


# 合并K个升序链表ok
def mergeKLists(lists) -> ListNode:
    pass


# 合并两个有序链表ok
def mergeTwoLists(l1: ListNode, l2: ListNode) -> ListNode:
    pass



# 删除链表的倒数第 N 个结点 ok
def removeNthFromEnd(head, n: int) -> ListNode:
    pass


# 全排列 nums: List[int]   res -> List[List[int]] ok
def permute(nums):
    pass


# 括号生成ok
def generateParenthesis(n: int):
    pass


# 电话号码的字母组合ok
def letterCombinations1(digits: str):
    pass
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
    pass
    return

# 最大数：一个数组能组成的最大数 ok
import functools
def largestNumber(nums) -> str:
    pass


# 实现stack ok
class MinStack:
    def __init__(self):
        pass
    def push(self, val: int) -> None:
        pass

    def pop(self) -> None:
        pass

    def top(self) -> int:
        pass

    def getMin(self) -> int:
        pass


# 移动零ok
def moveZeroes(nums) -> None:
    pass


# 轮转数组ok
def rotate(nums, k: int) -> None:
    pass

# 合并两个有序数组 num1初始m+n ok
def merge(nums1, m, nums2, n) -> None:
    pass

# 合并区间ok
def merge1(intervals: List[List[int]]) -> List[List[int]]:
    pass

# 最长公共前缀 str: List[str]  str可能'' ok
def longestCommonPrefix(strs):
    pass

# 盛最多水的容器 n>=2 ok
def maxArea(height) -> int:
    pass


# 两数之和ok
def twoSum(nums, target):
    pass

# topK partition ok
def topK_partition(arr, k):
    pass


# topK 自己实现堆ok
def topK_heap(arr, k):
    pass


# topk--python api ok
import heapq
def topK_api(arr, k):
    pass


# 基本二分ok
def binarySearch(arr, target):
    pass


# 二分左边界ok
def binarySearchLeftBoundry(arr, target):
    pass


# 二分右边界ok
def binarySearchRightBoundry(arr, target):
    pass

# 堆排ok
def heapSort(nums):
    pass


# 快排ok
import random
def quickSort(nums):
    pass



# 归并ok
def mergeSort(nums):
    pass


# 插入ok
def insertSort(nums):
    pass

# 选择ok
def selectSort(nums):
    pass


# 冒泡ok
def bubbleSort(nums):
    isSwap = True
    for end in range(len(nums) - 1, -1, -1):
        if not isSwap:
            break
        isSwap = False
        for i in range(end):
            if nums[i] > nums[i + 1]:
                nums[i], nums[i + 1] = nums[i + 1], nums[i]
                isSwap = True
# 三数之和为0
def threeSum(nums):
    pass


import numpy as np
def nms(preds, threshold):
    x1, y1, x2, y2, score = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3], preds[:, 4]
    keep = []
    order = np.argsort(score)[::-1]
    areas = (x2 - x1) * (y2 - y1)
    while(order.size > 0):
        maxIndex = order[0]
        keep.append(maxIndex)
        left_top_x = np.maximum(x1[maxIndex], x1[order[1:]])
        left_top_y = np.maximum(x2[maxIndex], x2[order[1:]])
        right_bottom_x = np.minimum(y1[maxIndex], y1[order[1:]])
        right_bottom_y = np.minimum(y2[maxIndex], y2[order[1:]])

        inner_w = np.maximum(right_bottom_x - left_top_x, 0.)
        inner_h = np.maximum(right_bottom_y - left_top_y, 0.)
        inner_area = inner_w * inner_h

        union_area = areas[maxIndex] + areas[order[1:]] - inner_area
        IOU = inner_area / union_area

        idx = np.where(IOU < threshold)[0]
        if idx.size <= 0:
            break

        order = order[idx + 1]

    return preds[keep]


def iou(preds, gt):
    left_top_x = np.maximum(preds[:, 0], gt[0])
    left_top_y = np.maximum(preds[:, 1], gt[1])
    right_bottom_x = np.minimum(preds[:, 2], gt[2])
    right_bottom_y = np.minimum(preds[:, 3], gt[3])

    inner_w = np.maximum(right_bottom_x - left_top_x, 0.)
    inner_h = np.maximum(right_bottom_y - left_top_y, 0.)
    inner_area = inner_w * inner_h

    union_area = (gt[2] - gt[0]) * (gt[3] - gt[1]) + (preds[:, 2] - preds[:, 0]) * (preds[:, 3] - preds[:, 1]) - inner_area

    IOU = inner_area / union_area

    maxIndex = np.argmax(IOU)[0]
    maxIOU = np.max(IOU)
    return IOU, maxIndex, maxIOU
































