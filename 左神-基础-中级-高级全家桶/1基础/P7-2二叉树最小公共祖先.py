class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


# --------------------------------- 1. 自己的递归解法 ---------------#
'''
核心就是 把路径存下来，然后比对。只不过左神是用map在遍历的时候，把 子-父 的关系存下来，
        那么从每个node的开始向上的父链都能找到。
        
    我这里是遍历的时候把从上到下的链存起来。左神的貌似更好理解一点。


'''

def lowestCommonAncestor1(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:

    res1 = findPath(root, p)
    res2 = findPath(root, q)
    i = 0
    while (i < len(res1) and i < len(res2) and res1[i] is res2[i]):
        i += 1

    return res1[i - 1]

# 寻找从root开始到指定结点的路径
def findPath(root, node):
    res = []
    if root is None:
        return res

    if root is node:
        res.append(node)
        return res

    res1 = findPath(root.left, node)
    res2 = findPath(root.right, node)

    if len(res1) != 0:
        res.append(root)
        res.extend(res1)
    if len(res2) != 0:
        res.append(root)
        res.extend(res2)

    return res

# -------------------------------------------------------------- #

# ------------------- 2. 特殊解法 ---------------#
'''

分析情况，公共祖先就两种情况：
(1)一个是另一个的祖先；
(2)有一个其他的公共祖先，向上汇聚的时候才能找到。


先碰到谁就返回谁：
（1）在一个子树，那先碰到的就是祖先；
（2）不在一个子树，在他的上级处理。
'''
# todo 需要记忆
def lowestCommonAncestor(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    if root is None or root is p or root is q:
        return root

    res1 = lowestCommonAncestor(root.left, p, q)
    res2 = lowestCommonAncestor(root.right, p, q)

    # todo 都不空，说明一个在左一个在右，当前就是祖先
    if res1 and res2:
        return root

    # todo 哪个不空返回哪个
    return res1 if res1 else res2


