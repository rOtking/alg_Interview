# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    # 递归
    def lowestCommonAncestor1(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        if p.val <= root.val <= q.val or q.val <= root.val <= p.val:
            return root

        elif p.val < root.val and q.val < root.val:
            return self.lowestCommonAncestor(root.left, p, q)

        else:
            return self.lowestCommonAncestor(root.right, p, q)

    # 迭代
    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        cur = root

        while (cur):
            if cur.val < p.val and cur.val < q.val:
                cur = cur.right
            elif cur.val > p.val and cur.val > q.val:
                cur = cur.left
            else:
                break
        return cur





if __name__ == '__main__':
    a = [1,2,3]
    b = [4, 5]
    for i,j in zip(a, b):
        print(i, j)




# ok
# todo BST的题一定要利用好BST的特性！！！！！继续理解。
