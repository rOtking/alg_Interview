# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    # todo 这就是DFS！！！
    def minDepth1(self, root: TreeNode) -> int:

        if root is None:
            return 0
        # todo 核心
        if root.left is None and root.right is None:
            return 1
        elif root.left is None:
            return self.minDepth(root.right) + 1
        elif root.right is None:
            return self.minDepth(root.left) + 1
        else:
            left_min = self.minDepth(root.left)
            right_min = self.minDepth(root.right)

            return min(left_min, right_min) + 1
    def minDepth(self, root: TreeNode) -> int:
        # 可以递归的做；但是
        # todo 求最短路径，显然是BFS的完美使用！BFS的思想，随着框架模拟一遍就很清晰了！
        if root is None:
            return 0
        queue = [root]

        depth = 1
        while(len(queue) != 0):
            for _ in range(len(queue)):
                cur = queue.pop(0)
                if cur.left is None and cur.right is None:
                    return depth

                if cur.left:
                    queue.append(cur.left)
                if cur.right:
                    queue.append(cur.right)
            depth += 1
        return depth

# ok
# todo BFS与DFS都能做
# todo 主要不要让只有一个节点的子树终止遍历，不是叶子节点是不能停下来的！