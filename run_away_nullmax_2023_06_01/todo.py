# todo

'''
1. 排序专题
'''

'''
冒泡 稳定 O(n^2)
流程：每轮从头到end，两两交换，保证每轮的end都是最大
key：确定end的边界

'''
def bubbleSort(nums):
    for end in range(len(nums) - 1, -1, -1):
        for i in range(end):   # end取不到，很好
            if nums[i] > nums[i + 1]:
                nums[i], nums[i + 1] = nums[i + 1], nums[i]
    return nums



'''
选择 不稳定 O(n^2)
流程：每轮从头到end遍历，更新max_index，与end最后交换，保证每轮结束时end都是最大值即可
key: 确定end与 max_index

6，5（1），2，5（2）---->5(2),5(1),2,6     5的顺序直接就乱了
'''
def selectSort(nums):
    for end in range(len(nums) - 1, -1, -1):
        max_index = end
        for i in range(end):
            if nums[i] > nums[max_index]:
                max_index = i
        nums[max_index], nums[end] = nums[end], nums[max_index]
    return nums

'''
插入 稳定 O(n^2)
流程：每轮都把当前位置插入到左边（已经排好序的）正确位置
key：到正确位置，直接break
'''
def insertSort(nums):
    for i in range(len(nums)):
        for j in range(i, -1, 0):
            if nums[j] < nums[j - 1]:
                nums[j], nums[j - 1] = nums[j - 1], nums[j]
            else:
                break
    return nums


'''
归并 稳定 O(nlogn)
流程：分治 + 递归, process得到[left,right]排好的结果, merge输入两个排好的list，输出一个大的排好的list
key：大递归思路
'''
def mergeSort(nums):
    def process(nums, left, right):
        if left >= right:
            return
        mid = left + ((right - left) >> 1)
        process(nums, left, mid)
        process(nums, mid + 1, right)
        merge(nums, left, right, mid)
        return

    def  merge(nums, left, right, mid):
        helps = []
        p1 = left
        p2 = mid + 1
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
        for i in range(left, right + 1):
            nums[i] = helps[i - left]       # todo 记得重新赋值！！
        return
    process(nums, 0, len(nums) - 1)
    return nums

'''
快速 不稳定 O(nlogn)
流程：分治 + 递归, process得到一个数字在nums上的正确位置，每个区域都这么做
key：大递归思路
'''
import random
def quickSort(nums):
    # todo process只排好一部分
    def process(nums, left, right):
        if left >= right:
            return
        tmp = left + int(random.random() * (right - left + 1))
        nums[right], nums[tmp] = nums[tmp], nums[right]
        small_right, big_left = partition(nums, left, right)
        process(nums, left, small_right)
        process(nums, big_left, right)

        return
    def partition(nums, left, right):
        '''
        通用模型
        小于...)等于... i 未知...(大于... pivot
        '''
        pivot = nums[right]
        small_right = left - 1
        big_left = right   # 为什么不-1，因为right-1不确定时大的
        i = left
        while(i < big_left):
            if nums[i] < pivot:
                small_right += 1
                # todo 易错点！!!
                nums[i], nums[small_right] = nums[small_right], nums[i]
                i += 1
            elif nums[i] == pivot:
                i += 1
            else:
                big_left -= 1
                nums[i], nums[big_left] = nums[big_left], nums[i]

        nums[i], nums[right] = nums[right], nums[i]
        big_left += 1    # todo 细节！
        return small_right, big_left
    process(nums, 0, len(nums) - 1)
    return nums


'''
堆 不稳定 O(nlogn)
流程：[0,end]构建大根堆 - 最大值放end位置 - [0,end-1]堆化 - 最大值放 end-1位置 - 直到结束
key：注意heapSize的变化
'''
def heapSort(nums):
    # 保证index之前的位置成为堆：堆尾插入一个，不断向上调整
    def heapInsert(nums, index):
        parent = int((index - 1) / 2)
        while(nums[index] > nums[parent]):
            nums[index], nums[parent] = nums[parent], nums[index]
            index = parent
            parent = int((index - 1) / 2)

    # 堆化 heapSize：从0位置开始多少个元素是堆
    # 只有index不满足，它的left和right都是满足的
    def heapify(nums, index, heapSize):
        left = 2 * index + 1
        while(left < heapSize):
            right = left + 1
            largest = right if right <= heapSize and nums[left] < nums[right] else left
            largest = index if nums[index] > nums[largest] else largest
            if largest == index:
                break
            else:
                nums[index], nums[largest] = nums[largest], nums[index]
                index = largest
                left = 2 * index + 1

    # (1)heapInsert初始化 O(NlogN)
    for i in range(len(nums)):
        heapInsert(nums, i)

    heapSize = len(nums) - 1
    while(heapSize > 0):
        nums[0], nums[heapSize] = nums[heapSize], nums[0]
        heapify(nums, 0, heapSize)
        heapSize -= 1
    return nums

# ------------------------------------------------------------------------------------------------------------------
'''
2. 二分查找
'''
def binarySearch(nums, target):
    left, right = 0, len(nums) - 1
    while(left <= right):
        mid = left + ((right - left) >> 1)
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# 找左边界    [1,2,2,2,2,6,7]
def binarySearchLeft(nums, target):
    left, right = 0, len(nums) - 1
    while(left <= right):
        mid = left + ((right - left) >> 1)
        if nums[mid] >= target:
            right = mid - 1
        else:
            left = mid + 1
    # todo 出来时 left == right + 1，有三种情况：
    #  1.right在index=0的左边，left在0位置,且num[left] != target--->target比所有数都小；
    #  2.right在len(nums) - 1,left超出索引，------->target比所有数都大；
    #  3.left在正常位置且nums[left] == target，那她就是左边界
    #  只要存在就一定是left
    if 0 < left < len(nums) - 1 and nums[left] == target:
        return left
    else:
        return -1

# ------------------------------------------------------------------------------------------------------------------
'''
3. BT
'''
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# 中序 递归
def inorderTraversal1(root: TreeNode):
    res = []
    if root is None:
        return root
    res.extend(inorderTraversal1(root.left))
    res.append(root.val)
    res.extend(inorderTraversal1(root.right))
    return res

# todo 中序最难 循环 有模版
def inorderTraversal2(root: TreeNode):
    stack = []
    res = []
    cur = root
    while(cur and stack):
        while(cur):
            stack.append(cur)
            cur = cur.left
        cur = stack.pop()
        res.append(cur.val)
        cur = cur.right
    return res

def preorderTraversal(root: TreeNode):
    res = []
    stack = [root] if root else []
    while(stack):
        cur = stack.pop()
        res.append(cur.val)
        if cur.right:
            stack.append(cur.right)
        if cur.left:
            stack.append(cur.left)
    return res


# 后序 迭代     中右左----倒----左右中
def postorderTraversal(root: TreeNode):
    res = []
    stack = [root] if root else []
    while(stack):
        cur = stack.pop()
        res.append(cur.val)
        if cur.left:
            stack.append(cur.left)
        if cur.right:
            stack.append(cur.right)
    return res[::-1]


# 验证二叉搜索树
# 所有left都小于cur，所有right都大于cur
def isValidBST(root: TreeNode) -> bool:
    def process(root):
        if root is None:
            return None
        left_res = process(root.left)
        right_res = process(root.right)

        leftOk = True if left_res is None or (left_res[0] and left_res[1] < root.val) else False
        rightOk = True if right_res is None or (right_res[0] and right_res[2] > root.val) else False
        max_value = right_res[1]
        min_value = left_res[2]
        return leftOk and rightOk, max_value, min_value

    return process(root)[0] if root else False


# 对称二叉树 pq问题
#            1
#        /        \
#      2           2
#    /   \       /  \
#  3     4     4     3
def isSymmetric(root: TreeNode) -> bool:
    # todo 检查 p和q是不是对称
    def check(p, q):
        if p is None and q is None:
            return True
        if p is None or q is None:
            return False

        res1 = check(p.left, q.right)
        res2 = check(p.right, q.left)
        return res1 and res2 and (p.val == q.val)

    if root is None:
        return True
    else:
        return check(root.left, root.right)


# 二叉树的层序遍历  -> List[List[int]]ok
# todo 自己的方法1，就是循环extend; 更好的是BFS！！！
def levelOrder(root: TreeNode):
    def process(root):
        if root is None:
            return []
        left_res = process(root.left)
        right_res = process(root.right)
        res = [[root.val]]
        i = 0
        while(i < len(left_res) and i < len(right_res)):
            tmp = left_res[i]
            tmp.extend(right_res[i])
            res.append(tmp)
            i += 1
        if i < len(left_res):
            res.extend(left_res[i:])
        if i < len(right_res):
            res.extend(right_res[i:])
        return res
    return process(root)

# 二叉树的最大深度
def maxDepth(root: TreeNode) -> int:
    if root is None:
        return 0
    ldepth = maxDepth(root.left)
    rdepth = maxDepth(root.right)
    return max(ldepth, rdepth) + 1

# 从前序与中序遍历序列构造二叉树
# 无重复数字！！考虑特性：前序的的第一个一定是 root-左树-右树，中序的一定是 左树-root-右树
def buildTree(preorder, inorder) -> TreeNode:
    if len(preorder) == 0:
        return None
    root_v = preorder[0]
    root_index = inorder.index(root_v)
    left = buildTree(preorder[1:root_index+1], inorder[:root_index])
    right = buildTree(preorder[root_index+1:], inorder[root_index:])
    root = TreeNode(root_v)
    root.left = left
    root.right = right
    return root


# 将有序数组转换为二叉搜索树 okok
def sortedArrayToBST(nums) -> TreeNode:
    if len(nums) == 0:
        return None

    mid = (len(nums) - 1) >> 1
    root = TreeNode(nums[mid])
    root.left = sortedArrayToBST(nums[:mid])
    root.right = sortedArrayToBST(nums[mid + 1:])
    return root

# 二叉树的最大路径和:可以从下往上
# todo 比较难，递归函数并不能直接解决问题，需要另一个变量来更新结果
def maxPathSum(root: TreeNode) -> int:
    # 以root为端点的单链最大
    def subMax(root):
        nonlocal res
        if root is None:
            return 0

        leftMax = max(subMax(root.left), 0)
        rightMax = max(subMax(root.right), 0)

        # todo 更新 必然 过每个节点的最大和
        res = root.val + leftMax + rightMax

        # todo 返回单链最大
        return root.val + max(leftMax, rightMax)

    if root is None:
        return 0
    res = -float('inf')
    _ = subMax(root)
    return res


# 二叉树的最近公共祖先 todo
# 所有节点的值都是唯一的。
# p、q 为不同节点且均存在于给定的二叉树中。
def lowestCommonAncestor(root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    if root is None or root is p or root is q:
        return root

    res1 = lowestCommonAncestor(root.left, p, q)
    res2 = lowestCommonAncestor(root.right, p, q)
    # todo 都不空，说明一个在左一个在右，当前就是祖先
    if res1 and res2:
        return root
    return res1 if res1 else res2

# 翻转二叉树ok
def invertTree(root: TreeNode) -> TreeNode:
    if root is None:
        return root
    left = invertTree(root.left)
    right = invertTree(root.right)
    root.right = left
    root.left = right
    return root

# 二叉树的直径ok
# 任意两个节点之间最长路径的 长度 。这条路径可能经过也可能不经过根节点 root
def diameterOfBinaryTree(root: TreeNode) -> int:
    # 单链最长长度
    def subMax(root):
        nonlocal res
        if root is None:
            return 0
        l_max = subMax(root.left)
        r_max = subMax(root.right)
        res = max(res, l_max + r_max + 1)
        return max(l_max, r_max) + 1
    res = 0
    _ = subMax(root)
    return res


# 路径总和 III:和为target的所有路径ok
# 路径方向必须是向下的（只能从父节点到子节点）
def pathSum(root: TreeNode, sum: int) -> int:
    # 以root为起点的个数
    def process(root, target):
        if root is None:
            if target == 0:
                return 1
            else:
                return 0
        return process(root.left, target - root.val) + process(root.right, target - root.val)

    if root is None:
        return 0

    res = pathSum(root.left, sum) + pathSum(root.right, sum) + \
          process(root.left, sum - root.val) + process(root.right, sum - root.val)
    return res

# 合并二叉树ok
# 结构相加
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

# 二叉树展开为链表
# 全都是right，先序遍历顺序
def flatten(root: TreeNode) -> None:
    # todo 只返回tail就可以
    def process(root):
        if root is None:
            return None
        if root.left is None and root.right is None:
            return root
        left_tail = process(root.left)
        right_tail = process(root.right)
        right_tmp_head = root.right
        root.right = root.left
        root.left = None
        left_tail.right = right_tmp_head
        right_tail.right = None
        return right_tail
    _ = process(root)
    return root

# ------------------------------------------------------------------------------------------------------------------
'''
4. list

核心是用好dummy指针
'''
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

# 删除链表的倒数第 N 个结点 ok
# todo 快慢指针
def removeNthFromEnd(head, n: int) -> ListNode:
    dummy = ListNode(0)
    dummy.next = head
    slow, fast = dummy, dummy
    for _ in range(n):
        fast = fast.next
    while(fast.next):
        slow = slow.next
        fast = fast.next
    # 此时fast为倒数第1，slow为倒数n+1
    toDel = slow.next
    slow.next = toDel.next
    toDel.next = None
    dummy.next = None
    return head

# 合并两个有序链表ok
def mergeTwoLists(l1: ListNode, l2: ListNode) -> ListNode:
    dummy = ListNode(0)
    pre = dummy
    p1 = l1
    p2 = l2
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


# 合并K个升序链表ok
# todo 二分的拆解为 两两合并！
def mergeKLists(lists) -> ListNode:
    # todo 合并[left,right]上的所有list
    def process(lists, left, right):
        if left == right:
            return lists[left]
        mid = left + ((right - left) >> 1)
        head1 = process(lists, left, mid)
        head2 = process(lists, mid + 1, right)
        return merge(head1, head2)
    def merge(head1, head2):
        dummy = ListNode(0)
        p1, p2 = head1, head2
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


class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random

# 复制带随机指针的链表ok
def copyRandomList(head: 'Node') -> 'Node':
    pass



if __name__ == '__main__':
    pass

