'''
记住核心性质即可
'''
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
''''
总结：

BST、满二叉树、平衡二叉树都能用递归套路解决，只要注意递归函数的返回值要设计成你做判断需要的全量信息即可。
一般自己额外定义一个递归的process函数，可能需要多个返回值。

完全二叉树的判断通过l与r的信息是不方便直接进行判断的。


树型DP的问题都能递归的解决！

'''
# ------------------------------- 1. 搜索二叉树BST ---------------------#
'''
BST没有重复值。

判断方法：
1.BST就是任一个子树都满足  左<根<右
2.BST的中序遍历一定是升序。

具体看刷题内容吧，没啥多说的。还是递归的好做！
'''
# todo 验证ok！

def isValidBST(root: TreeNode) -> bool:
    def process(root):
        if root is None:
            return None

        isValid = True
        # todo 这个赋值很精髓！
        minValue = root.val
        maxValue = root.val

        res1 = process(root.left)
        res2 = process(root.right)

        # left不空
        if res1:
            if res2:
                isValid = True if res1[0] and res2[0] and (res1[2] < root.val < res2[1]) else False
                minValue = min(res1[1], root.val, res2[1])
                maxValue = max(res1[2], root.val, res2[2])
            else:
                isValid = True if res1[0] and res1[2] < root.val else False
                minValue = min(res1[1], root.val)
                maxValue = max(res1[2], root.val)
        # left空
        else:
            if res2:
                isValid = True if res2[0] and root.val < res2[1] else False
                minValue = min(res2[1], root.val)
                maxValue = max(res2[2], root.val)
            else:
                pass

        return isValid, minValue, maxValue

    if root is None:
        return True
    res = process(root)
    return res[0]






# ---------------------------------------------------------------------#

# ------------------------------- 2. 完全二叉树 ---------------------#
'''
堆那里说过，从上到下，从左到右，依次填满。

判断方法：
BFS遍历，判断
（1）任一个node，有右无左直接false；
满足的（1）的前提下：（2）若遇到了第一个左右不双全的node，包括没有左右的叶结点与只有左节点，那么
                    剩下的所有node都必须是叶子，那就是完全BT，否则不是！

'''

def checkCBT(root):
    if root is None:
        return True

    queue = [root]

    mustLeaf = False     # 剩下的是否必须都是叶子

    while(len(queue) != 0):
        # todo 这层for循环其实可以不加！
        for _ in range(len(queue)):
            cur = queue.pop(0)
            if cur.right and cur.left is None:
                return False

            if mustLeaf and (cur.left or cur.right):
                return False

            if cur.left:
                queue.append(cur.left)
            if cur.right:
                queue.append(cur.right)
            # todo 左右不双全，进行标记  单独写，方便理解
            if cur.left is None or cur.right is None:
                mustLeaf = True

    return True
'''
p1 = TreeNode()
p2 = TreeNode()
p3 = TreeNode()
p4 = TreeNode()
p5 = TreeNode()

p1.left = p2
p1.right = p3
p2.left = p4

a = checkCBT(p1)
print(a)    # True

p3.right = p5
a = checkCBT(p1)
print(a)    # Fasle
'''

# todo 没有太多的例子，但看上去是对的！



# ---------------------------------------------------------------------#

# ------------------------------- 3. 判断满二叉树 ---------------------#
'''
就是全满的特殊类型，如果深度是n，则满足：

结点个数 = 2^n - 1

普通方法：先遍历个数，在求深度，验证即可。

递归：满 = 左满 + 右满 + 左右高度一样
'''
def isFullBT(root):
    # 返回 是不是满 + 高度
    def process(root):
        if root is None:
            return True, 0

        res1 = process(root.left)
        res2 = process(root.right)

        isFull = False
        if res1[0] and res2[0] and res1[1] == res2[1]:
            isFull = True

        return isFull, max(res1[1], res2[1]) + 1

    return process(root)[0]

# todo ok 验证通过！
'''
p1 = TreeNode()
p2 = TreeNode()
p3 = TreeNode()
p4 = TreeNode()
p5 = TreeNode()

p1.left = p2
p1.right = p3

r = isFullBT(p1)
print(r)

p2.left = p4
r = isFullBT(p1)
print(r)

p2.right = p5
r = isFullBT(p1)
print(r)
'''

# ---------------------------------------------------------------------#
# ------------------------------- 4. 判平衡很二叉树 ---------------------#
'''
任何子树的左与右，深度差不超过1,即<=1

递归！

'''
# todo 验证ok！

def isBalanced(root: TreeNode) -> bool:
    # 返回是不是 + 深度
    def process(root):
        if root is None:
            return True, 0

        res1 = process(root.left)
        res2 = process(root.right)

        isOk = False
        if res1[0] and res2[0] and (-2 < res1[1] - res2[1] < 2):
            isOk = True

        height = max(res1[1], res2[1]) + 1

        return isOk, height

    return process(root)[0]



# ---------------------------------------------------------------------#