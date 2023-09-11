
'''
递归遍历最简单
'''
# ----------------------- 1.递归序 ------------------ #
def f(head):
    if head.left is None and head.right is None:
        return head
    # 1.第一次来到本节点

    f(head.left)
    # 2.第二次来到这个结点

    f(head.right)
    # 3.第三次来到这个结点
'''
总会3次到达一个结点，即使什么都没做！
          1
        /   \ 
       2     3
      / \   / \ 
     4  5  6   7

递归序：
1,
  2,4,4,4,
  2,5,5,5,
  2,
1,
  3,6,6,6,
  3,7,7,7,
  3,
1 

前中后就是在第几次print即可，可就是同一个数字递归序出现3次，
    前序就是第一次出现打印，后面的不要了
    中序   第二次
    后续   第三次

'''

# ------------------------------------------------ #

# ----------------------- 2.非递归 ------------------ #
'''
任何递归都能改为非递归！

用stack来自己压栈

1.前序流程：头左右

先把根放入stack；

（1）stack pop一个结点cur；
（2）处理（打印）cur；
（3）（如果有）先右后左 孩子push进stack；
（4）重复上面3步。


2.后序遍历：左右头      也就是前序'版本"头右左"的逆序啊
先把根放入stack1,辅助stack2；

（1）stack1 pop一个结点cur；
（2）处理（打印）cur，存入stack逆序；
（3）（如果有）先左后右 孩子push进stack；
（4）重复上面3步。
stack2依次pop即可


难点
3.中序:左中右
stack
（1）每个子树所有左边界依次进stack；
（2）pop cur；
（3）如果cur有right，push进stack
（4）重复
先记忆把

实质是：整个树被左边界分解掉了！
               +
              /
             O    +
           /   \ /
          O   + O   +
         / \ / / \ /
        O   O O   O
       /   / /   /
      +   + +   +
所有结点由左边界组成：
左中（右）
     |
     左中（右）
          |
          左中（右）
               |
              ...
'''
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    # 迭代法
    # todo 前序ok了
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        if root is None:
            return res
        cur = root
        stack = [cur]
        while(len(stack) != 0):
            cur = stack.pop()
            res.append(cur.val)
            if cur.right:
                stack.append(cur.right)
            if cur.left:
                stack.append(cur.left)
        return res

    # todo 后序OK了
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        if root is None:
            return res

        stack1 = [root]
        stack2 = []
        while (len(stack1) != 0):
            cur = stack1.pop()
            stack2.append(cur.val)
            if cur.left:
                stack1.append(cur.left)
            if cur.right:
                stack1.append(cur.right)

        while (len(stack2) != 0):
            res.append(stack2.pop())

        return res

    # todo 重点中序！！！！！！
    #  cur向左走到头，输出自己，加入right
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        if root is None:
            return res
        stack = []
        cur = root
        # todo 这个条件与loop中的分支很关键！
        while (len(stack) != 0 or cur):
            if cur:
                stack.append(cur)
                cur = cur.left
            # 左边界走到头
            else:
                cur = stack.pop()
                res.append(cur.val)
                cur = cur.right

        return res


